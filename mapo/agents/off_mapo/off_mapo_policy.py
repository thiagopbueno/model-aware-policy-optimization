"""MAPOTFPolicy with DDPG and TD3 tricks."""
import tensorflow as tf
from tensorflow import keras
from gym.spaces import Box

from ray.rllib.policy import build_tf_policy, TFPolicy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.annotations import override

from mapo.models.policy import build_deterministic_policy
from mapo.models.q_function import build_continuous_q_function


def _build_critic_targets(policy, batch_tensors):
    rewards, dones, next_obs = (
        batch_tensors[SampleBatch.REWARDS],
        batch_tensors[SampleBatch.DONES],
        batch_tensors[SampleBatch.NEXT_OBS],
    )
    gamma = policy.config["gamma"]
    target_q_model = policy.target_q_model
    next_action = policy.target_policy_model(next_obs)
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
    next_q_values = tf.squeeze(target_q_model([next_obs, next_action]))
    if policy.config["twin_q"]:
        target_twin_q_model = policy.target_twin_q_model
        next_q_values = tf.math.minimum(
            next_q_values, tf.squeeze(target_twin_q_model([next_obs, next_action]))
        )
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
    q_pred = tf.squeeze(policy.q_model([obs, actions]))
    q_stats = {
        "q_mean": tf.reduce_mean(q_pred),
        "q_max": tf.reduce_max(q_pred),
        "q_min": tf.reduce_min(q_pred),
    }
    policy.loss_stats.update(q_stats)
    critic_loss = q_loss_criterion(q_pred, target_q_values)
    if policy.config["twin_q"]:
        twin_q_pred = tf.squeeze(policy.twin_q_model([obs, actions]))
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
    q_model = policy.q_model
    policy_model = policy.policy_model
    policy_action_value = q_model([obs, policy_model(obs)])
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
    """Get the default configuration for OffMAPOTFPolicy."""
    # pylint: disable=cyclic-import
    from mapo.agents.off_mapo.off_mapo import DEFAULT_CONFIG

    return DEFAULT_CONFIG


def extra_loss_fetches(policy, _):
    """Add stats computed along with the loss function."""
    return policy.loss_stats


def actor_critic_gradients(policy, *_):
    """Create compute gradients ops using separate optimizers."""
    # pylint: disable=protected-access
    actor_grads = policy._actor_optimizer.get_gradients(
        policy.actor_loss, policy.policy_model.variables
    )

    critic_variables = policy.q_model.variables
    if policy.config["twin_q"]:
        critic_variables += policy.twin_q_model.variables
    critic_grads = policy._critic_optimizer.get_gradients(
        policy.critic_loss, critic_variables
    )
    # Save these for later use in build_apply_op
    policy._actor_grads_and_vars = list(zip(actor_grads, policy.policy_model.variables))
    policy._critic_grads_and_vars = list(zip(critic_grads, critic_variables))
    return policy._actor_grads_and_vars + policy._critic_grads_and_vars


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


def create_separate_optimizers(policy, obs_space, action_space, config):
    """Initialize optimizers and global step for update operations."""
    # pylint: disable=unused-argument,protected-access
    policy._actor_optimizer = keras.optimizers.Adam(learning_rate=config["actor_lr"])
    policy._critic_optimizer = keras.optimizers.Adam(learning_rate=config["critic_lr"])
    policy.global_step = tf.Variable(0, trainable=False)


def copy_targets(policy, obs_space, action_space, config):
    """Copy parameters from original models to target models."""
    # pylint: disable=unused-argument
    policy.target_init = policy.build_update_targets_op(tau=1)
    policy.get_session().run(policy.target_init)


def build_action_sampler(policy, input_dict, action_space, config):
    """Add exploration noise when not evaluating the policy."""
    deterministic_actions = policy.policy_model(input_dict[SampleBatch.CUR_OBS])
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

    return tf.cond(
        policy.evaluating,
        true_fn=lambda: deterministic_actions,
        false_fn=make_exploration_actions,
    )


def build_actor_critic_models(policy, input_dict, obs_space, action_space, config):
    """Construct actor and critic keras models, and return actor action tensor."""
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

    def make_actor():
        return build_deterministic_policy(
            obs_space, action_space, config["actor_model"]
        )

    def make_critic():
        return build_continuous_q_function(
            obs_space, action_space, config["critic_model"]
        )

    policy.policy_model, policy.target_policy_model = make_actor(), make_actor()
    policy.q_model, policy.target_q_model = make_critic(), make_critic()
    policy.main_variables = policy.policy_model.variables + policy.q_model.variables
    policy.target_variables = (
        policy.target_policy_model.variables + policy.target_q_model.variables
    )
    if config["twin_q"]:
        policy.twin_q_model, policy.target_twin_q_model = make_critic(), make_critic()
        policy.main_variables += policy.twin_q_model.variables
        policy.target_variables += policy.target_twin_q_model.variables

    return build_action_sampler(policy, input_dict, action_space, config), None


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
        main_variables = self.main_variables
        target_variables = self.target_variables
        update_target_expr = [
            target_var.assign(tau * var + (1.0 - tau) * target_var)
            for var, target_var in zip(main_variables, target_variables)
        ]
        return tf.group(*update_target_expr)


class DelayedUpdatesMixin:  # pylint: disable=too-few-public-methods
    """
    Sets ops to update models with different frequencies.

    This mixin is needed since `build_tf_policy` does not have an argument to
    override `TFPolicy.build_apply_op`.
    """

    @override(TFPolicy)
    def build_apply_op(self, *_):
        """
        Update actor and critic models with different frequencies.

        For policy gradient, update policy net one time v.s. update critic net
        `policy_delay` time(s). Also use `policy_delay` for target networks update.
        """
        with tf.control_dependencies([self.global_step.assign_add(1)]):
            # Critic updates
            critic_op = self._critic_optimizer.apply_gradients(
                self._critic_grads_and_vars
            )
            # Actor updates
            should_apply_actor_opt = tf.equal(
                tf.math.mod(self.global_step, self.config["policy_delay"]), 0
            )

            def make_actor_apply_op():
                return self._actor_optimizer.apply_gradients(self._actor_grads_and_vars)

            with tf.control_dependencies([critic_op]):
                actor_op = tf.cond(
                    should_apply_actor_opt,
                    true_fn=make_actor_apply_op,
                    false_fn=tf.no_op,
                )
            # increment global step & apply ops
            apply_ops = tf.group(actor_op, critic_op)
            with tf.control_dependencies([apply_ops]):
                update_targets_op = tf.cond(
                    should_apply_actor_opt,
                    true_fn=self.build_update_targets_op,
                    false_fn=tf.no_op,
                )
            return tf.group(apply_ops, update_targets_op)


OffMAPOTFPolicy = build_tf_policy(
    name="OffMAPOTFPolicy",
    loss_fn=build_actor_critic_losses,
    get_default_config=get_default_config,
    stats_fn=extra_loss_fetches,
    optimizer_fn=lambda *_: None,
    gradients_fn=actor_critic_gradients,
    extra_action_feed_fn=extra_action_feed_fn,
    before_init=setup_early_mixins,
    before_loss_init=create_separate_optimizers,
    after_init=copy_targets,
    make_action_sampler=build_actor_critic_models,
    mixins=[DelayedUpdatesMixin, TargetUpdatesMixin, ExplorationStateMixin],
    obs_include_prev_action_reward=False,
)
