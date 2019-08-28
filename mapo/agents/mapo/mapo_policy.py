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
from ray.rllib.models.model import restore_original_dimensions
from ray.rllib.evaluation.postprocessing import compute_advantages

import mapo.agents.mapo.losses as losses


AgentComponents = namedtuple("AgentComponents", "dynamics critic actor")


def build_mapo_losses(policy, batch_tensors):
    """Contruct dynamics (MLE/PG-aware), critic (Fitted Q) and actor (MADPG) losses."""
    # Can't alter the original batch_tensors
    # RLlib tracks which keys have been accessed in batch_tensors. Then, at runtime it
    # feeds each corresponding value its respective sample batch array. If we alter
    # a dictionary field by restoring dimensions, the new value might be a tuple or
    # dict, which can't be fed an array when calling the session later.
    batch_tensors = {key: batch_tensors[key] for key in batch_tensors}
    for key in [SampleBatch.CUR_OBS, SampleBatch.NEXT_OBS]:
        batch_tensors[key] = restore_original_dimensions(
            batch_tensors[key], policy.observation_space
        )
    model, config = policy.model, policy.config
    env = _global_registry.get(ENV_CREATOR, config["env"])(config["env_config"])

    dynamics_fetches = {}
    if config["use_true_dynamics"]:
        dynamics_loss = None
    elif config["model_loss"] == "pga":
        dynamics_loss, dynamics_fetches = losses.dynamics_pga_loss(
            batch_tensors, model, env, config
        )
    else:
        dynamics_loss = losses.dynamics_mle_loss(batch_tensors, model)
    critic_loss, critic_fetches = losses.critic_return_loss(batch_tensors, model)
    actor_loss = losses.actor_model_aware_loss(batch_tensors, model, env, config)

    policy.loss_stats = {}
    if not config["use_true_dynamics"]:
        policy.loss_stats["dynamics_loss"] = dynamics_loss
    policy.loss_stats["critic_loss"] = critic_loss
    policy.loss_stats["actor_loss"] = actor_loss
    policy.loss_stats.update(critic_fetches)
    policy.loss_stats.update(dynamics_fetches)

    policy.mapo_losses = AgentComponents(
        dynamics=dynamics_loss, critic=critic_loss, actor=actor_loss
    )
    mapo_loss = critic_loss + actor_loss
    if not config["use_true_dynamics"]:
        mapo_loss += dynamics_loss
    return mapo_loss


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


def _var_name(var):
    # pylint: disable=missing-docstring
    return "/".join(var.name.split("/")[1:])


def extra_grad_fetches(policy, _):
    """Add stats computed along with the compute gradients function."""
    if not policy.config["debug"]:
        return {}
    all_grads_and_vars = policy.all_grads_and_vars

    def grad_and_vars_stats(grads_and_vars):
        for grad, var in grads_and_vars:
            tf.compat.v1.summary.histogram(_var_name(grad), grad)
            tf.compat.v1.summary.histogram(_var_name(var), var)
            tf.compat.v1.summary.scalar(_var_name(grad) + "/norm", tf.norm(grad))

    if not policy.config["use_true_dynamics"]:
        grad_and_vars_stats(all_grads_and_vars.dynamics)

        dynamics_model = policy.model.models["dynamics"]
        log_stddev = dynamics_model.log_stddev
        abs_log_stddev = tf.abs(log_stddev)
        min_abs_log_stddev = tf.reduce_min(abs_log_stddev)
        with tf.name_scope("dynamics_model/log_stddev"):
            tf.compat.v1.summary.scalar("min_abs_log_stddev", min_abs_log_stddev)

    grad_and_vars_stats(all_grads_and_vars.critic)
    grad_and_vars_stats(all_grads_and_vars.actor)

    merged = tf.compat.v1.summary.merge_all()
    return {"summaries": merged}


def create_separate_optimizers(policy, config):
    """Initialize optimizers and global step for update operations."""
    # pylint: disable=unused-argument
    actor_optimizer_config = {
        "class_name": config["actor_optimizer"],
        "config": {
            "learning_rate": config["actor_lr"],
            "name": "actor" + config["actor_optimizer"],
        },
    }
    critic_optimizer_config = {
        "class_name": config["critic_optimizer"],
        "config": {
            "learning_rate": config["critic_lr"],
            "name": "critic" + config["critic_optimizer"],
        },
    }
    actor_optimizer = keras.optimizers.get(actor_optimizer_config)
    critic_optimizer = keras.optimizers.get(critic_optimizer_config)

    if config["use_true_dynamics"]:
        dynamics_optimizer = None
    else:
        dynamics_optimizer_config = {
            "class_name": config["dynamics_optimizer"],
            "config": {
                "learning_rate": config["dynamics_lr"],
                "name": "dynamics" + config["dynamics_optimizer"],
            },
        }
        dynamics_optimizer = keras.optimizers.get(dynamics_optimizer_config)

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
        grads = optim.get_gradients(loss, variables)
        return list(zip(grads, variables))

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


def apply_op_n_times(sgd_iter, apply_op):
    """Apply op `sgd_iter` times in a while loop."""
    # pylint: disable=invalid-name
    i0 = tf.constant(0)

    def cond(i):
        return tf.less(i, sgd_iter)

    def body(i):
        app_op = apply_op()
        with tf.control_dependencies([app_op]):
            inc_op = tf.add(i, 1)
        return inc_op

    return tf.while_loop(cond, body, [i0])


def apply_gradients_n_times(policy, optimizer, grads_and_vars):
    """Update critic, dynamics, and actor for a number of iterations."""
    # pylint: disable=unused-argument
    config = policy.config
    global_step = policy.global_step
    grads_and_vars = policy.all_grads_and_vars
    model = policy.model

    # Update critic
    def update_critic_fn():
        apply_grads = optimizer.critic.apply_gradients(grads_and_vars.critic)
        with tf.control_dependencies([apply_grads]):
            should_update_target = tf.equal(
                tf.math.mod(
                    optimizer.critic.iterations, config["critic_target_update_freq"]
                ),
                0,
            )

            def reset_target_fn():
                return tf.group(
                    [
                        target.assign(main)
                        for main, target in zip(
                            model.models["q_net"].variables,
                            model.target_models["q_net"].variables,
                        )
                    ]
                )

            def soft_update_fn():
                update_target_expr = [
                    target.assign(config["tau"] * main + (1.0 - config["tau"]) * target)
                    for main, target in zip(
                        model.models["q_net"].variables,
                        model.target_models["q_net"].variables,
                    )
                ]
                update_all_target_variables = tf.group(*update_target_expr)
                return update_all_target_variables

            def update_target_fn():
                return tf.cond(
                    critic_target_should_reset,
                    true_fn=reset_target_fn,
                    false_fn=lambda: tf.cond(
                        critic_prep_done, true_fn=soft_update_fn, false_fn=tf.no_op
                    ),
                )

            update_target = tf.cond(
                should_update_target, true_fn=update_target_fn, false_fn=tf.no_op
            )
        return tf.group(apply_grads, update_target)

    with tf.control_dependencies([global_step.assign_add(1)]):
        critic_target_should_reset = tf.equal(global_step, config["critic_prep_steps"])
        critic_prep_done = tf.greater(global_step, config["critic_prep_steps"])
        critic_op = apply_op_n_times(config["critic_sgd_iter"], update_critic_fn)

    # Update Dynamics
    def dynamics_fn():
        if config["use_true_dynamics"]:
            dynamics_op = tf.no_op()
        else:
            dynamics_op = apply_op_n_times(
                config["dynamics_sgd_iter"],
                lambda: optimizer.dynamics.apply_gradients(grads_and_vars.dynamics),
            )
        return tf.group(dynamics_op)

    with tf.control_dependencies([critic_op]):
        dynamics_op = tf.cond(critic_prep_done, true_fn=dynamics_fn, false_fn=tf.no_op)

    # Update Actor
    def update_actor_fn():
        apply_actor_op = optimizer.actor.apply_gradients(grads_and_vars.actor)
        with tf.control_dependencies([apply_actor_op]):
            should_update_target = tf.equal(
                tf.math.mod(
                    optimizer.actor.iterations, config["actor_target_update_freq"]
                ),
                0,
            )

            def update_target_fn():
                update_target_expr = [
                    target.assign(config["tau"] * main + (1.0 - config["tau"]) * target)
                    for main, target in zip(
                        model.models["policy"].variables,
                        model.target_models["policy"].variables,
                    )
                ]
                return tf.group(*update_target_expr)

            update_actor_target = tf.cond(
                should_update_target, true_fn=update_target_fn, false_fn=tf.no_op
            )
        return tf.group(apply_actor_op, update_actor_target)

    with tf.control_dependencies([dynamics_op]):
        actor_op = tf.cond(critic_prep_done, true_fn=update_actor_fn, false_fn=tf.no_op)

    return tf.group(critic_op, dynamics_op, actor_op)


def apply_gradients_with_delays(policy, optimizer, grads_and_vars):
    """
    Update actor and critic models with different frequencies.

    For policy gradient, update policy net one time v.s. update critic net
    `actor_delay` time(s). Also use `actor_delay` for target networks update.
    """
    # pylint: disable=unused-argument
    global_step, config = policy.global_step, policy.config
    grads_and_vars = policy.all_grads_and_vars
    with tf.control_dependencies([global_step.assign_add(1)]):
        # Critic updates
        critic_op = tf.cond(
            tf.equal(tf.math.mod(global_step, config["critic_delay"]), 0),
            true_fn=lambda: optimizer.critic.apply_gradients(grads_and_vars.critic),
            false_fn=tf.no_op,
        )

        # Dynamics updates
        with tf.control_dependencies([critic_op]):
            if config["use_true_dynamics"]:
                dynamics_op = tf.no_op()
            else:
                dynamics_op = tf.cond(
                    tf.equal(tf.math.mod(global_step, config["dynamics_delay"]), 0),
                    true_fn=lambda: optimizer.dynamics.apply_gradients(
                        grads_and_vars.dynamics
                    ),
                    false_fn=tf.no_op,
                )

        # Actor updates
        with tf.control_dependencies([dynamics_op]):
            actor_op = tf.cond(
                tf.equal(tf.math.mod(global_step, config["actor_delay"]), 0),
                true_fn=lambda: optimizer.actor.apply_gradients(grads_and_vars.actor),
                false_fn=tf.no_op,
            )

        return tf.group(dynamics_op, critic_op, actor_op)


def apply_gradients_fn(policy, optimizer, grads_and_vars):
    """Choose between 'sgd_iter' and 'delayed' strategies for applying gradients."""
    if policy.config["apply_gradients"] == "sgd_iter":
        return apply_gradients_n_times(policy, optimizer, grads_and_vars)
    if policy.config["apply_gradients"] == "delayed":
        return apply_gradients_with_delays(policy, optimizer, grads_and_vars)
    raise ValueError(
        "Invalid apply gradients strategy '{}'. "
        "Try one of [sgd_iter, delayed]".format(policy.config["apply_gradients"])
    )


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
        target_networks=True,
        twin_q=False,
    )


def main_actor_output(policy, model, input_dict, obs_space, action_space, config):
    """Simply use the deterministic actor output as the action."""
    # pylint: disable=too-many-arguments,unused-argument
    return (
        model.compute_actions(
            restore_original_dimensions(input_dict[SampleBatch.CUR_OBS], obs_space)
        ),
        None,
    )


MAPOTFPolicy = build_tf_policy(
    name="MAPOTFPolicy",
    loss_fn=build_mapo_losses,
    get_default_config=get_default_config,
    postprocess_fn=compute_return,
    stats_fn=extra_loss_fetches,
    grad_stats_fn=extra_grad_fetches,
    optimizer_fn=create_separate_optimizers,
    gradients_fn=compute_separate_gradients,
    apply_gradients_fn=apply_gradients_fn,
    make_model=build_mapo_network,
    action_sampler_fn=main_actor_output,
)
