"""MAPO Tensorflow Policy."""
from collections import namedtuple

import tensorflow as tf
from tensorflow import keras
from gym.spaces import Box

from ray.tune.registry import ENV_CREATOR, _global_registry
from ray.rllib.policy import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.evaluation.postprocessing import compute_advantages

import mapo.agents.mapo.losses as losses


AgentComponents = namedtuple("AgentComponents", "dynamics critic actor")


def build_mapo_losses(policy, batch_tensors):
    """Contruct dynamics (MLE/PG-aware), critic (Fitted Q) and actor (MADPG) losses."""
    model, config = policy.model, policy.config
    env = _global_registry.get(ENV_CREATOR, config["env"])(config["env_config"])
    actor_loss = losses.actor_model_aware_loss(batch_tensors, model, env, config)
    if config["use_true_dynamics"]:
        dynamics_loss = 0
    elif config["model_loss"] == "pga":
        dynamics_loss = losses.dynamics_pga_loss(
            batch_tensors, model, actor_loss, config
        )
    else:
        dynamics_loss = losses.dynamics_mle_loss(batch_tensors, model)
    critic_loss = losses.critic_return_loss(batch_tensors, model)
    policy.loss_stats = {}
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


def extra_loss_fetches(policy, _):
    """Add stats computed along with the loss function."""
    return policy.loss_stats


def create_separate_optimizers(policy, config):
    """Initialize optimizers and global step for update operations."""
    # pylint: disable=unused-argument
    actor_optimizer = keras.optimizers.Adam(learning_rate=config["actor_lr"])
    critic_optimizer = keras.optimizers.Adam(learning_rate=config["critic_lr"])
    if config["use_true_dynamics"]:
        dynamics_optimizer = None
    else:
        dynamics_optimizer = keras.optimizers.Adam(learning_rate=config["dynamics_lr"])
    policy.global_step = tf.Variable(0, trainable=False)
    return AgentComponents(
        dynamics=dynamics_optimizer, critic=critic_optimizer, actor=actor_optimizer
    )


def compute_separate_gradients(policy, optimizer, loss):
    """Create compute gradients ops using separate optimizers."""
    # pylint: disable=unused-argument
    config = policy.config
    dynamics_loss, critic_loss, actor_loss = (
        policy.mapo_losses.dynamics,
        policy.mapo_losses.critic,
        policy.mapo_losses.actor,
    )

    def grads_and_vars(loss, optim, variables):
        return list(zip(optim.get_gradients(loss, variables), variables))

    if config["use_true_dynamics"]:
        dynamics_grads_and_vars = []
    else:
        dynamics_grads_and_vars = grads_and_vars(
            dynamics_loss, optimizer.dynamics, policy.model.dynamics_variables
        )
    critic_grads_and_vars = grads_and_vars(
        critic_loss, optimizer.critic, policy.model.critic_variables
    )
    actor_grads_and_vars = grads_and_vars(
        actor_loss, optimizer.actor, policy.model.actor_variables
    )

    policy.all_grads_and_vars = AgentComponents(
        dynamics=None if config["use_true_dynamics"] else dynamics_grads_and_vars,
        critic=critic_grads_and_vars,
        actor=actor_grads_and_vars,
    )
    return dynamics_grads_and_vars + critic_grads_and_vars + actor_grads_and_vars


def apply_gradients_with_delays(policy, optimizer, grads_and_vars):
    """
    Update actor and critic models with different frequencies.

    For policy gradient, update policy net one time v.s. update critic net
    `actor_delay` time(s). Also use `actor_delay` for target networks update.
    """
    # pylint: disable=unused-argument
    dynamics_grads_and_vars, critic_grads_and_vars, actor_grads_and_vars = (
        policy.all_grads_and_vars.dynamics,
        policy.all_grads_and_vars.critic,
        policy.all_grads_and_vars.actor,
    )
    with tf.control_dependencies([policy.global_step.assign_add(1)]):
        # Dynamics updates
        if policy.config["use_true_dynamics"]:
            dynamics_op = tf.no_op()
        else:
            dynamics_op = optimizer.dynamics.apply_gradients(dynamics_grads_and_vars)
        # Critic updates
        should_apply_critic_opt = tf.equal(
            tf.math.mod(policy.global_step, policy.config["critic_delay"]), 0
        )
        with tf.control_dependencies([dynamics_op]):
            critic_op = tf.cond(
                should_apply_critic_opt,
                true_fn=lambda: optimizer.critic.apply_gradients(critic_grads_and_vars),
                false_fn=tf.no_op,
            )
        # Actor updates
        should_apply_actor_opt = tf.equal(
            tf.math.mod(policy.global_step, policy.config["actor_delay"]), 0
        )
        with tf.control_dependencies([critic_op]):
            actor_op = tf.cond(
                should_apply_actor_opt,
                true_fn=lambda: optimizer.actor.apply_gradients(actor_grads_and_vars),
                false_fn=tf.no_op,
            )
        return tf.group(dynamics_op, critic_op, actor_op)


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
        create_dynamics=not config["use_true_dynamics"],
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
    stats_fn=extra_loss_fetches,
    optimizer_fn=create_separate_optimizers,
    gradients_fn=compute_separate_gradients,
    apply_gradients_fn=apply_gradients_with_delays,
    make_model=build_mapo_network,
    action_sampler_fn=main_actor_output,
)
