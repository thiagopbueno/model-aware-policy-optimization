"""MAPO Tensorflow Policy."""
import tensorflow as tf
from tensorflow import keras
from gym.spaces import Box

from ray.rllib.policy import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing


def _build_dynamics_mle_loss(batch_tensors, model):
    obs, actions, next_obs = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
        batch_tensors[SampleBatch.NEXT_OBS],
    )
    return -tf.reduce_mean(model.compute_states_log_prob(obs, actions, next_obs))


def _build_critic_loss(batch_tensors, model):
    # Fitted Q loss (using trajectory returns)
    obs, actions, returns = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
        batch_tensors[Postprocessing.ADVANTAGES],
    )
    return keras.losses.mean_squared_error(
        model.compute_q_values(obs, actions), returns
    )


def _build_actor_loss(batch_tensors, model, config):
    obs = batch_tensors[SampleBatch.CUR_OBS]
    gamma = config["gamma"]
    n_samples = config["branching_factor"]
    policy_action = model.compute_actions(obs)
    sampled_next_state, next_state_log_prob = model.compute_log_prob_sampled(
        obs, policy_action, (n_samples,)
    )
    next_state_value = tf.stop_gradient(model.compute_state_values(sampled_next_state))
    model_aware_policy_loss = tf.reduce_mean(
        gamma * tf.reduce_mean(next_state_log_prob * next_state_value, axis=0)
    )
    return model_aware_policy_loss


def build_mapo_losses(policy, batch_tensors):
    """Contruct dynamics (MLE/PG-aware), critic (Fitted Q) and actor (MADPG) losses."""
    policy.loss_stats = {}
    if policy.config["model_loss"] == "mle":
        policy.dynamics_loss = _build_dynamics_mle_loss(batch_tensors, policy.model)
    elif policy.config["model_loss"] == "pg-aware":
        raise NotImplementedError
    else:
        raise ValueError(
            "Unknown model_loss '{}' (try 'mle' or 'pg-aware')".format(
                policy.config["model_loss"]
            )
        )
    policy.critic_loss = _build_critic_loss(batch_tensors, policy.model)
    policy.actor_loss = _build_actor_loss(batch_tensors, policy.model, policy.config)
    policy.loss_stats["dynamics_loss"] = policy.dynamics_loss
    policy.loss_stats["critic_loss"] = policy.critic_loss
    policy.loss_stats["actor_loss"] = policy.actor_loss
    return policy.dynamics_loss + policy.critic_loss + policy.actor_loss


def get_default_config():
    """Get the default configuration for MAPOTFPolicy."""
    from mapo.agents.mapo.mapo import DEFAULT_CONFIG  # pylint: disable=cyclic-import

    return DEFAULT_CONFIG


def compute_return(policy, sample_batch, other_agent_batches=None, episode=None):
    """Add trajectory return to sample_batch."""
    # pylint: disable=unused-argument
    return compute_advantages(sample_batch, 0.0, policy.config["gamma"], use_gae=False)


def create_separate_optimizers(policy, config):
    """Initialize optimizers and global step for update operations."""
    # pylint: disable=protected-access
    policy._actor_optimizer = keras.optimizers.Adam(learning_rate=config["actor_lr"])
    policy._critic_optimizer = keras.optimizers.Adam(learning_rate=config["critic_lr"])
    policy._dynamics_optimizer = keras.optimizers.Adam(
        learning_rate=config["dynamics_lr"]
    )
    policy.global_step = tf.Variable(0, trainable=False)


def compute_separate_gradients(policy, optimizer, loss):
    """Create compute gradients ops using separate optimizers."""
    # pylint: disable=protected-access,unused-argument
    actor_variables = policy.model.actor_variables
    critic_variables = policy.model.critic_variables
    dynamics_variables = policy.model.dynamics_variables
    actor_grads = policy._actor_optimizer.get_gradients(
        policy.actor_loss, actor_variables
    )
    critic_grads = policy._critic_optimizer.get_gradients(
        policy.critic_loss, critic_variables
    )
    dynamics_grads = policy._dynamics_optimizer.get_gradients(
        policy.dynamics_loss, dynamics_variables
    )
    # Save these for later use in build_apply_op
    policy._actor_grads_and_vars = list(zip(actor_grads, actor_variables))
    policy._critic_grads_and_vars = list(zip(critic_grads, critic_variables))
    policy._dynamics_grads_and_vars = list(zip(dynamics_grads, dynamics_variables))
    return (
        policy._actor_grads_and_vars
        + policy._critic_grads_and_vars
        + policy._dynamics_grads_and_vars
    )


def apply_gradients_with_delays(policy, optimizer, grads_and_vars):
    """
    Update actor and critic models with different frequencies.

    For policy gradient, update policy net one time v.s. update critic net
    `policy_delay` time(s). Also use `policy_delay` for target networks update.
    """
    # pylint: disable=protected-access,unused-argument
    with tf.control_dependencies([policy.global_step.assign_add(1)]):
        # Critic updates
        critic_op = policy._critic_optimizer.apply_gradients(
            policy._critic_grads_and_vars
        )
        # Actor updates
        should_apply_actor_opt = tf.equal(
            tf.math.mod(policy.global_step, policy.config["policy_delay"]), 0
        )

        def make_actor_apply_op():
            return policy._actor_optimizer.apply_gradients(policy._actor_grads_and_vars)

        with tf.control_dependencies([critic_op]):
            actor_op = tf.cond(
                should_apply_actor_opt, true_fn=make_actor_apply_op, false_fn=tf.no_op
            )
        return tf.group(actor_op, critic_op)


def build_mapo_network(policy, obs_space, action_space, config):
    """Construct MAPOModelV2 networks."""
    # pylint: disable=unused-argument
    if not isinstance(action_space, Box):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for MAPO.".format(action_space)
        )
    if len(action_space.shape) > 1:
        raise UnsupportedSpaceException(
            "Action space has multiple dimensions {}.".format(action_space.shape)
            + "Consider reshaping this into a single dimension, using a Tuple action"
            "space, or the multi-agent API."
        )
    return ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        1,
        model_config=config["model"],
        framework="tf",
        name="mapo_model",
    )


def main_actor_output(policy, model, input_dict, obs_space, action_space, config):
    """Simply use the deterministic actor output as the action."""
    # pylint: disable=too-many-arguments,unused-argument
    return model.compute_actions(input_dict[SampleBatch.CUR_OBS]), None


MAPOTFPolicy = build_tf_policy(
    name="MAPOTFPolicy",
    loss_fn=build_mapo_losses,
    get_default_config=get_default_config,
    postprocess_fn=compute_return,
    optimizer_fn=create_separate_optimizers,
    gradients_fn=compute_separate_gradients,
    apply_gradients_fn=apply_gradients_with_delays,
    make_model=build_mapo_network,
    action_sampler_fn=main_actor_output,
)
