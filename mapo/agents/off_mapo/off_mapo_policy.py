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
            next_action,
            tf.constant(policy.action_space.low),
            tf.constant(policy.action_space.high),
        )
    next_q_values = target_q_model([next_obs, next_action])
    if policy.config["twin_q"]:
        target_twin_q_model = policy.target_twin_q_model
        next_q_values = tf.math.minimum(
            next_q_values, target_twin_q_model([next_obs, next_action])
        )
    bootstrapped = rewards + gamma * tf.squeeze(next_q_values)
    # Do not bootstrap if the state is terminal
    return tf.compat.v1.where(dones, x=rewards, y=bootstrapped)


def _build_critic_loss(policy, batch_tensors):
    obs, actions = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
    )
    target_q_values = _build_critic_targets(policy, batch_tensors)
    q_loss_criterion = keras.losses.MeanSquaredError()
    q_pred = policy.q_model([obs, actions])
    q_stats = {
        "q_mean": tf.reduce_mean(q_pred),
        "q_max": tf.reduce_max(q_pred),
        "q_min": tf.reduce_min(q_pred),
    }
    policy.loss_stats.update(q_stats)
    critic_loss = q_loss_criterion(q_pred, target_q_values)
    if policy.config["twin_q"]:
        twin_q_pred = policy.twin_q_model([obs, actions])
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
    return -tf.reduce_mean(q_model([obs, policy_model(obs)]))


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


def check_action_space(policy, obs_space, action_space, config):
    """Check if the action space is suited to DPG."""
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


def build_actor_critic_models(policy, input_dict, obs_space, action_space, config):
    """Construct actor and critic keras models, and return actor action tensor."""
    policy.q_model = build_continuous_q_function(obs_space, action_space, config)
    policy.policy_model = build_deterministic_policy(obs_space, action_space, config)
    policy.target_q_model = build_continuous_q_function(obs_space, action_space, config)
    policy.target_policy_model = build_deterministic_policy(
        obs_space, action_space, config
    )
    policy.main_variables = policy.q_model.variables + policy.policy_model.variables
    policy.target_variables = (
        policy.target_q_model.variables + policy.target_policy_model.variables
    )
    if config["twin_q"]:
        policy.twin_q_model = build_continuous_q_function(
            obs_space, action_space, config
        )
        policy.target_twin_q_model = build_continuous_q_function(
            obs_space, action_space, config
        )
        policy.main_variables += policy.twin_q_model.variables
        policy.target_variables += policy.target_twin_q_model.variables

    actions = policy.policy_model(input_dict[SampleBatch.CUR_OBS])
    return actions, None


class TargetUpdatesMixin:
    """Adds methods to build ops that update target networks."""

    # pylint: disable=too-few-public-methods
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
        # Actor updates
        should_apply_actor_opt = tf.equal(
            tf.math.mod(self.global_step, self.config["policy_delay"]), 0
        )

        def make_actor_apply_op():
            app_grad = self._actor_optimizer.apply_gradients(self._actor_grads_and_vars)
            with tf.control_dependencies([app_grad]):
                update_target_op = self.build_update_targets_op()
            return tf.group(app_grad, update_target_op)

        actor_op = tf.cond(
            should_apply_actor_opt, true_fn=make_actor_apply_op, false_fn=tf.no_op
        )
        # Critic updates
        critic_op = self._critic_optimizer.apply_gradients(self._critic_grads_and_vars)
        # increment global step & apply ops
        with tf.control_dependencies([self.global_step.assign_add(1)]):
            return tf.group(actor_op, critic_op)


OffMAPOTFPolicy = build_tf_policy(
    name="OffMAPOTFPolicy",
    loss_fn=build_actor_critic_losses,
    get_default_config=get_default_config,
    stats_fn=extra_loss_fetches,
    optimizer_fn=lambda *_: None,
    gradients_fn=actor_critic_gradients,
    before_init=check_action_space,
    before_loss_init=create_separate_optimizers,
    after_init=copy_targets,
    make_action_sampler=build_actor_critic_models,
    mixins=[DelayedUpdatesMixin, TargetUpdatesMixin],
    obs_include_prev_action_reward=False,
)
