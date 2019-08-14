"""MAPOTFPolicy with DDPG and TD3 tricks."""
import tensorflow as tf
from tensorflow import keras
from gym.spaces import Box

from ray.rllib.policy import build_tf_policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.error import UnsupportedSpaceException

from mapo.agents.mapo.mapo_policy import (
    AgentComponents,
    _build_dynamics_mle_loss,
    extra_loss_fetches,
    create_separate_optimizers,
    compute_separate_gradients,
)


def _build_critic_targets(batch_tensors, model, action_space, config):
    rewards, dones, next_obs = (
        batch_tensors[SampleBatch.REWARDS],
        batch_tensors[SampleBatch.DONES],
        batch_tensors[SampleBatch.NEXT_OBS],
    )
    gamma = config["gamma"]
    next_action = model.compute_actions(next_obs, target=True)
    if config["smooth_target_policy"]:
        epsilon = tf.random.normal(tf.shape(next_action), stddev=config["target_noise"])
        epsilon = tf.clip_by_value(
            epsilon, -config["target_noise_clip"], config["target_noise_clip"]
        )
        next_action = next_action + epsilon
        next_action = tf.clip_by_value(next_action, action_space.low, action_space.high)
    next_q_values = tf.squeeze(
        model.compute_q_values(next_obs, next_action, target=True)
    )
    if config["twin_q"]:
        twin_q_values = model.compute_twin_q_values(next_obs, next_action, target=True)
        next_q_values = tf.math.minimum(next_q_values, tf.squeeze(twin_q_values))
    # Do not bootstrap if the state is terminal
    bootstrapped = rewards + gamma * next_q_values
    return tf.compat.v2.where(dones, x=rewards, y=bootstrapped)


def _build_critic_loss(batch_tensors, model, action_space, config):
    obs, actions = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
    )
    target_q_values = _build_critic_targets(batch_tensors, model, action_space, config)
    q_loss_criterion = keras.losses.MeanSquaredError()
    q_pred = tf.squeeze(model.compute_q_values(obs, actions))
    fetches = {
        "q_mean": tf.reduce_mean(q_pred),
        "q_max": tf.reduce_max(q_pred),
        "q_min": tf.reduce_min(q_pred),
    }
    critic_loss = q_loss_criterion(q_pred, target_q_values)
    if config["twin_q"]:
        twin_q_pred = tf.squeeze(model.compute_twin_q_values(obs, actions))
        fetches.update(
            {
                "twin_q_mean": tf.reduce_mean(twin_q_pred),
                "twin_q_max": tf.reduce_max(twin_q_pred),
                "twin_q_min": tf.reduce_min(twin_q_pred),
            }
        )
        twin_q_loss = q_loss_criterion(twin_q_pred, target_q_values)
        critic_loss += twin_q_loss
    return critic_loss, fetches


def _build_actor_loss(batch_tensors, model, config):
    obs = batch_tensors[SampleBatch.CUR_OBS]
    gamma = config["gamma"]
    n_samples = config["branching_factor"]

    policy_action = model.compute_actions(obs)
    sampled_next_state, next_state_log_prob = model.compute_log_prob_sampled(
        obs, policy_action, (n_samples,)
    )
    next_state_value = tf.stop_gradient(model.compute_state_values(sampled_next_state))
    # later: add reward function gradient before gamma
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
    critic_loss, critic_stats = _build_critic_loss(
        batch_tensors, policy.model, policy.action_space, policy.config
    )
    actor_loss = _build_actor_loss(batch_tensors, policy.model, policy.config)
    policy.loss_stats.update(critic_stats)
    policy.loss_stats["dynamics_loss"] = dynamics_loss
    policy.loss_stats["critic_loss"] = critic_loss
    policy.loss_stats["actor_loss"] = actor_loss
    policy.mapo_losses = AgentComponents(
        dynamics=dynamics_loss, critic=critic_loss, actor=actor_loss
    )
    return dynamics_loss + critic_loss + actor_loss


def get_default_config():
    """Get the default configuration for OffMAPOTFPolicy."""
    # pylint: disable=cyclic-import
    from mapo.agents.mapo.off_mapo import DEFAULT_CONFIG

    return DEFAULT_CONFIG


def ignore_timeout_termination(
    policy, sample_batch, other_agent_batches=None, episode=None
):
    """
    Set last done to false if episode length is greater or equal to the horizon.

    Runs the risk of ignoring other non-timeout terminations that coincide with the
    preset horizon.
    """
    # pylint: disable=unused-argument
    horizon = policy.config["horizon"]
    if episode and horizon and episode.length >= horizon:
        sample_batch[SampleBatch.DONES][-1] = False
    return sample_batch


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
        apply_ops = tf.group(dynamics_op, critic_op, actor_op)
        with tf.control_dependencies([apply_ops]):
            update_targets_op = tf.cond(
                should_apply_actor_opt,
                true_fn=policy.build_update_targets_op,
                false_fn=tf.no_op,
            )
        return tf.group(apply_ops, update_targets_op)


def extra_action_feed_fn(policy):
    """Add exploration status to compute_actions feed dict."""
    return {
        policy.evaluating: policy.evaluation,
        policy.pure_exploration_phase: policy.uniform_random,
    }


def setup_early_mixins(policy, obs_space, action_space, config):
    """Initialize early stateful mixins."""
    # pylint: disable=unused-argument
    ExplorationStateMixin.__init__(policy, config["evaluate"])


def copy_targets(policy, obs_space, action_space, config):
    """Copy parameters from original models to target models."""
    # pylint: disable=unused-argument
    policy.target_init = policy.build_update_targets_op(tau=1)
    policy.get_session().run(policy.target_init)


def build_action_sampler(policy, model, input_dict, obs_space, action_space, config):
    """Add exploration noise when not evaluating the policy."""
    # pylint: disable=too-many-arguments,unused-argument
    deterministic_actions = model.compute_actions(input_dict[SampleBatch.CUR_OBS])
    policy.evaluating = tf.placeholder(tf.bool, shape=[])
    policy.pure_exploration_phase = tf.placeholder(tf.bool, shape=[])

    def make_noisy_actions():
        # shape of deterministic_actions is [None, dim_action]
        if config["exploration_noise_type"] == "gaussian":
            # add IID Gaussian noise for exploration, TD3-style
            normal_sample = tf.random.normal(
                tf.shape(deterministic_actions),
                stddev=config["exploration_gaussian_sigma"],
            )
            stochastic_actions = tf.clip_by_value(
                deterministic_actions + normal_sample,
                action_space.low,
                action_space.high,
            )
        elif config["exploration_noise_type"] == "ou":
            # add OU noise for exploration, DDPG-style
            exploration_sample = tf.get_variable(
                name="ornstein_uhlenbeck",
                dtype=tf.float32,
                initializer=lambda: tf.zeros(action_space.shape),
                trainable=False,
            )
            normal_sample = tf.random.normal(action_space.shape, mean=0.0, stddev=1.0)
            ou_new = (
                -config["exploration_ou_theta"] * exploration_sample
                + config["exploration_ou_sigma"] * normal_sample
            )
            exploration_value = tf.assign_add(exploration_sample, ou_new)
            base_scale = config["exploration_ou_noise_scale"]
            noise = (
                base_scale * exploration_value * (action_space.high - action_space.low)
            )
            stochastic_actions = tf.clip_by_value(
                deterministic_actions + noise, action_space.low, action_space.high
            )
        else:
            raise ValueError(
                "Unknown noise type '{}' (try 'ou' or 'gaussian')".format(
                    config["exploration_noise_type"]
                )
            )
        return stochastic_actions

    def make_uniform_random_actions():
        # pure random exploration option
        return tf.random.uniform(
            tf.shape(deterministic_actions),
            minval=action_space.low,
            maxval=action_space.high,
        )

    def make_exploration_actions():
        return tf.cond(
            policy.pure_exploration_phase,
            true_fn=make_uniform_random_actions,
            false_fn=make_noisy_actions,
        )

    actions = tf.cond(
        policy.evaluating,
        true_fn=lambda: deterministic_actions,
        false_fn=make_exploration_actions,
    )
    return actions, None


def build_mapo_network(policy, obs_space, action_space, config):
    """Construct MAPOModelV2 networks with target actor and critic."""
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
        target_networks=True,
        twin_q=config["twin_q"],
    )


class ExplorationStateMixin:  # pylint: disable=too-few-public-methods
    """Adds method to toggle pure exploration phase."""

    def __init__(self, evaluate):
        self.uniform_random = False
        self.evaluation = evaluate

    def set_pure_exploration_phase(self, pure_exploration):
        """Set flag for computing uniform random actions."""
        self.uniform_random = pure_exploration

    def evaluate(self, evaluate):
        """Set flag for computing deterministic actions."""
        self.evaluation = evaluate


class TargetUpdatesMixin:  # pylint: disable=too-few-public-methods
    """Adds methods to build ops that update target networks."""

    def build_update_targets_op(self, tau=None):
        """Build op to update target networks."""
        tau = tau or self.config["tau"]
        update_target_expr = [
            target.assign(tau * main + (1.0 - tau) * target)
            for main, target in self.model.main_and_target_variables
        ]
        return tf.group(*update_target_expr)


OffMAPOTFPolicy = build_tf_policy(
    name="OffMAPOTFPolicy",
    loss_fn=build_mapo_losses,
    get_default_config=get_default_config,
    postprocess_fn=ignore_timeout_termination,
    stats_fn=extra_loss_fetches,
    optimizer_fn=create_separate_optimizers,
    gradients_fn=compute_separate_gradients,
    apply_gradients_fn=apply_gradients_with_delays,
    extra_action_feed_fn=extra_action_feed_fn,
    before_init=setup_early_mixins,
    after_init=copy_targets,
    make_model=build_mapo_network,
    action_sampler_fn=build_action_sampler,
    mixins=[TargetUpdatesMixin, ExplorationStateMixin],
    obs_include_prev_action_reward=False,
)
