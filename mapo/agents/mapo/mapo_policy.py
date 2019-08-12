"""MAPO Tensorflow Policy."""
import tensorflow as tf
from tensorflow import keras
from gym.spaces import Box

from ray.rllib.policy import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing


def build_actor_critic_losses(policy, batch_tensors):
    """Contruct actor (DPG) and critic (Fitted Q) tf losses."""
    # Fitted Q loss (using trajectory returns)
    obs, actions, returns = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
        batch_tensors[Postprocessing.ADVANTAGES],
    )
    policy.critic_loss = keras.losses.mean_squared_error(
        policy.model.get_q_values(obs, actions), returns
    )
    # DPG loss
    policy_action = policy.model.get_actions(obs)
    policy_action_value = policy.model.get_q_values(obs, policy_action)
    policy.actor_loss = -tf.reduce_mean(policy_action_value)
    return policy.actor_loss + policy.critic_loss


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
    policy.global_step = tf.Variable(0, trainable=False)


def actor_critic_gradients(policy, *_):
    """Create compute gradients ops using separate optimizers."""
    # pylint: disable=protected-access
    actor_variables = policy.model.actor_variables
    critic_variables = policy.model.critic_variables
    actor_grads = policy._actor_optimizer.get_gradients(
        policy.actor_loss, actor_variables
    )
    critic_grads = policy._critic_optimizer.get_gradients(
        policy.critic_loss, critic_variables
    )
    # Save these for later use in build_apply_op
    policy._actor_grads_and_vars = list(zip(actor_grads, actor_variables))
    policy._critic_grads_and_vars = list(zip(critic_grads, critic_variables))
    return policy._actor_grads_and_vars + policy._critic_grads_and_vars


def apply_gradients_with_delays(policy, *_):
    """
    Update actor and critic models with different frequencies.

    For policy gradient, update policy net one time v.s. update critic net
    `policy_delay` time(s). Also use `policy_delay` for target networks update.
    """
    # pylint: disable=protected-access
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


def build_actor_critic_network(policy, obs_space, action_space, config):
    """Construct actor and critic keras models, wrapped in the ModelV2 interface."""
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
    return model.get_actions(input_dict[SampleBatch.CUR_OBS]), None


MAPOTFPolicy = build_tf_policy(
    name="MAPOTFPolicy",
    loss_fn=build_actor_critic_losses,
    get_default_config=get_default_config,
    postprocess_fn=compute_return,
    optimizer_fn=create_separate_optimizers,
    gradients_fn=actor_critic_gradients,
    apply_gradients_fn=apply_gradients_with_delays,
    make_model=build_actor_critic_network,
    action_sampler_fn=main_actor_output,
)
