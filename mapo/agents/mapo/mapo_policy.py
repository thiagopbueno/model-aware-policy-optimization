"""MAPO Tensorflow Policy."""
from collections import namedtuple

import tensorflow as tf
from tensorflow import keras
from gym.spaces import Box

from ray.rllib.policy import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing


AgentComponents = namedtuple("AgentComponents", "dynamics critic actor")


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
        dynamics_loss = _build_dynamics_mle_loss(batch_tensors, policy.model)
    elif policy.config["model_loss"] == "pg-aware":
        raise NotImplementedError
    else:
        raise ValueError(
            "Unknown model_loss '{}' (try 'mle' or 'pg-aware')".format(
                policy.config["model_loss"]
            )
        )
    critic_loss = _build_critic_loss(batch_tensors, policy.model)
    actor_loss = _build_actor_loss(batch_tensors, policy.model, policy.config)
    policy.loss_stats["dynamics_loss"] = dynamics_loss
    policy.loss_stats["critic_loss"] = critic_loss
    policy.loss_stats["actor_loss"] = actor_loss
    policy.mapo_losses = AgentComponents(
        dynamics=dynamics_loss, critic=critic_loss, actor=actor_loss
    )
    return dynamics_loss + critic_loss + actor_loss


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
    # pylint: disable=unused-argument
    actor_optimizer = keras.optimizers.Adam(learning_rate=config["actor_lr"])
    critic_optimizer = keras.optimizers.Adam(learning_rate=config["critic_lr"])
    dynamics_optimizer = keras.optimizers.Adam(learning_rate=config["dynamics_lr"])
    policy.global_step = tf.Variable(0, trainable=False)
    return AgentComponents(
        dynamics=dynamics_optimizer, critic=critic_optimizer, actor=actor_optimizer
    )


def compute_separate_gradients(policy, optimizer, loss):
    """Create compute gradients ops using separate optimizers."""
    # pylint: disable=unused-argument
    dynamics_loss, critic_loss, actor_loss = (
        policy.mapo_losses.dynamics,
        policy.mapo_losses.critic,
        policy.mapo_losses.actor,
    )
    dynamics_variables = policy.model.dynamics_variables
    critic_variables = policy.model.critic_variables
    actor_variables = policy.model.actor_variables

    dynamics_grads = optimizer.dynamics.get_gradients(dynamics_loss, dynamics_variables)
    critic_grads = optimizer.critic.get_gradients(critic_loss, critic_variables)
    actor_grads = optimizer.actor.get_gradients(actor_loss, actor_variables)
    dynamics_grads_and_vars = list(zip(dynamics_grads, dynamics_variables))
    critic_grads_and_vars = list(zip(critic_grads, critic_variables))
    actor_grads_and_vars = list(zip(actor_grads, actor_variables))
    policy.all_grads_and_vars = AgentComponents(
        dynamics=dynamics_grads_and_vars,
        critic=critic_grads_and_vars,
        actor=actor_grads_and_vars,
    )
    return dynamics_grads_and_vars + critic_grads_and_vars + actor_grads_and_vars


def apply_gradients_with_delays(policy, optimizer, grads_and_vars):
    """
    Update actor and critic models with different frequencies.

    For policy gradient, update policy net one time v.s. update critic net
    `policy_delay` time(s). Also use `policy_delay` for target networks update.
    """
    # pylint: disable=unused-argument
    dynamics_grads_and_vars, critic_grads_and_vars, actor_grads_and_vars = (
        policy.all_grads_and_vars.dynamics,
        policy.all_grads_and_vars.critic,
        policy.all_grads_and_vars.actor,
    )
    with tf.control_dependencies([policy.global_step.assign_add(1)]):
        # Dynamics updates
        dynamics_op = optimizer.dynamics.apply_gradients(dynamics_grads_and_vars)
        # Critic updates
        should_apply_critic_opt = tf.equal(
            tf.math.mod(policy.global_step, policy.config["critic_delay"]), 0
        )

        def make_critic_apply_op():
            return optimizer.critic.apply_gradients(critic_grads_and_vars)

        with tf.control_dependencies([dynamics_op]):
            critic_op = tf.cond(
                should_apply_critic_opt, true_fn=make_critic_apply_op, false_fn=tf.no_op
            )
        # Actor updates
        should_apply_actor_opt = tf.equal(
            tf.math.mod(policy.global_step, policy.config["policy_delay"]), 0
        )

        def make_actor_apply_op():
            return optimizer.actor.apply_gradients(actor_grads_and_vars)

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
