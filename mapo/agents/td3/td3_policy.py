"""TD3TFPolicy implementing TD3 tricks on top of DDPG."""
import tensorflow as tf
from tensorflow import keras
from gym.spaces import Box

from ray.rllib.policy import build_tf_policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.utils.error import UnsupportedSpaceException


def _build_critic_targets(policy, batch_tensors):
    rewards, dones, next_obs = (
        batch_tensors[SampleBatch.REWARDS],
        batch_tensors[SampleBatch.DONES],
        batch_tensors[SampleBatch.NEXT_OBS],
    )
    gamma = policy.config["gamma"]
    target_model = policy.target_model
    next_action = policy.target_model.get_actions(next_obs)
    if policy.config["smooth_target_policy"]:
        epsilon = tf.random.normal(
            tf.shape(next_action), stddev=policy.config["target_noise"]
        )
        epsilon = tf.clip_by_value(
            epsilon,
            -policy.config["target_noise_clip"],
            policy.config["target_noise_clip"],
        )
        next_action = next_action + epsilon
        next_action = tf.clip_by_value(
            next_action, policy.action_space.low, policy.action_space.high
        )
    next_q_values = tf.squeeze(target_model.get_q_values(next_obs, next_action))
    if policy.config["twin_q"]:
        twin_q_values = target_model.get_twin_q_values(next_obs, next_action)
        next_q_values = tf.math.minimum(next_q_values, tf.squeeze(twin_q_values))
    # Do not bootstrap if the state is terminal
    bootstrapped = rewards + gamma * next_q_values
    return tf.compat.v2.where(dones, x=rewards, y=bootstrapped)


def _build_critic_loss(policy, batch_tensors):
    obs, actions = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
    )
    target_q_values = _build_critic_targets(policy, batch_tensors)
    q_loss_criterion = keras.losses.MeanSquaredError()
    q_pred = tf.squeeze(policy.model.get_q_values(obs, actions))
    q_stats = {
        "q_mean": tf.reduce_mean(q_pred),
        "q_max": tf.reduce_max(q_pred),
        "q_min": tf.reduce_min(q_pred),
    }
    policy.loss_stats.update(q_stats)
    critic_loss = q_loss_criterion(q_pred, target_q_values)
    if policy.config["twin_q"]:
        twin_q_pred = tf.squeeze(policy.model.get_twin_q_values(obs, actions))
        twin_q_stats = {
            "twin_q_mean": tf.reduce_mean(twin_q_pred),
            "twin_q_max": tf.reduce_max(twin_q_pred),
            "twin_q_min": tf.reduce_min(twin_q_pred),
        }
        policy.loss_stats.update(twin_q_stats)
        twin_q_loss = q_loss_criterion(twin_q_pred, target_q_values)
        critic_loss += twin_q_loss
    return critic_loss


def _build_actor_loss(policy, batch_tensors):
    obs = batch_tensors[SampleBatch.CUR_OBS]
    policy_action = policy.model.get_actions(obs)
    policy_action_value = policy.model.get_q_values(obs, policy_action)
    policy.loss_stats["qpi_mean"] = tf.reduce_mean(policy_action_value)
    return -tf.reduce_mean(policy_action_value)


def build_actor_critic_losses(policy, batch_tensors):
    """Contruct actor (DPG) and critic (Fitted Q) losses."""
    policy.loss_stats = {}
    policy.critic_loss = _build_critic_loss(policy, batch_tensors)
    policy.actor_loss = _build_actor_loss(policy, batch_tensors)
    policy.loss_stats["critic_loss"] = policy.critic_loss
    policy.loss_stats["actor_loss"] = policy.actor_loss
    return policy.actor_loss + policy.critic_loss


def get_default_config():
    """Get the default configuration for TD3TFPolicy."""
    # pylint: disable=cyclic-import
    from mapo.agents.td3.td3 import DEFAULT_CONFIG

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


def extra_loss_fetches(policy, _):
    """Add stats computed along with the loss function."""
    return policy.loss_stats


def create_separate_optimizers(policy, config):
    """Initialize optimizers and global step for update operations."""
    # pylint: disable=protected-access
    actor_optimizer_config = {
        "class_name": config["actor_optimizer"],
        "config": {"learning_rate": config["actor_lr"]},
    }
    critic_optimizer_config = {
        "class_name": config["critic_optimizer"],
        "config": {"learning_rate": config["critic_lr"]},
    }
    policy._actor_optimizer = keras.optimizers.get(actor_optimizer_config)
    policy._critic_optimizer = keras.optimizers.get(critic_optimizer_config)
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
    `actor_delay` time(s). Also use `actor_delay` for target networks update.
    """
    # pylint: disable=protected-access
    with tf.control_dependencies([policy.global_step.assign_add(1)]):
        # Critic updates
        critic_op = policy._critic_optimizer.apply_gradients(
            policy._critic_grads_and_vars
        )
        # Actor updates
        should_apply_actor_opt = tf.equal(
            tf.math.mod(policy.global_step, policy.config["actor_delay"]), 0
        )

        def make_actor_apply_op():
            return policy._actor_optimizer.apply_gradients(policy._actor_grads_and_vars)

        with tf.control_dependencies([critic_op]):
            actor_op = tf.cond(
                should_apply_actor_opt, true_fn=make_actor_apply_op, false_fn=tf.no_op
            )
        apply_ops = tf.group(actor_op, critic_op)
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
    deterministic_actions = model.get_actions(input_dict[SampleBatch.CUR_OBS])
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
        uniform_random_actions = tf.random.uniform(tf.shape(deterministic_actions))
        # rescale uniform random actions according to action range
        tf_range = tf.constant(action_space.high - action_space.low)
        tf_low = tf.constant(action_space.low)
        uniform_random_actions = uniform_random_actions * tf_range + tf_low
        return uniform_random_actions

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


def build_actor_critic_network(policy, obs_space, action_space, config):
    """Construct actor and critic keras models, wrapped in the ModelV2 interface."""
    if not isinstance(action_space, Box):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for TD3.".format(action_space)
        )
    if len(action_space.shape) > 1:
        raise UnsupportedSpaceException(
            "Action space has multiple dimensions {}.".format(action_space.shape)
            + "Consider reshaping this into a single dimension, using a Tuple action"
            "space, or the multi-agent API."
        )
    policy.model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        1,
        model_config=config["model"],
        framework="tf",
        name="td3_model",
        twin_q=config["twin_q"],
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space,
        action_space,
        1,
        model_config=config["model"],
        framework="tf",
        name="td3_model",
        twin_q=config["twin_q"],
    )

    return policy.model


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
        main_variables = self.model.variables()
        target_variables = self.target_model.variables()
        update_target_expr = [
            target_var.assign(tau * var + (1.0 - tau) * target_var)
            for var, target_var in zip(main_variables, target_variables)
        ]
        return tf.group(*update_target_expr)


TD3TFPolicy = build_tf_policy(
    name="TD3TFPolicy",
    loss_fn=build_actor_critic_losses,
    get_default_config=get_default_config,
    postprocess_fn=ignore_timeout_termination,
    stats_fn=extra_loss_fetches,
    optimizer_fn=create_separate_optimizers,
    gradients_fn=actor_critic_gradients,
    apply_gradients_fn=apply_gradients_with_delays,
    extra_action_feed_fn=extra_action_feed_fn,
    before_init=setup_early_mixins,
    after_init=copy_targets,
    make_model=build_actor_critic_network,
    action_sampler_fn=build_action_sampler,
    mixins=[TargetUpdatesMixin, ExplorationStateMixin],
    obs_include_prev_action_reward=False,
)
