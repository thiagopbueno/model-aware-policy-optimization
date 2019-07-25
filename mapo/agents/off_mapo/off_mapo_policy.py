"""MAPOTFPolicy with DDPG and TD3 tricks."""
import tensorflow as tf
from tensorflow import keras
from gym.spaces import Box

from ray.rllib.policy import build_tf_policy, TFPolicy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.annotations import override
from ray.rllib.policy.sample_batch import SampleBatch

from mapo.models.policy import build_deterministic_policy
from mapo.models.q_function import build_continuous_q_function


def build_actor_critic_losses(policy, batch_tensors):
    """Contruct actor (DPG) and critic (Fitted Q) losses."""
    obs, actions, rewards, dones, next_obs = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
        batch_tensors[SampleBatch.REWARDS],
        batch_tensors[SampleBatch.DONES],
        batch_tensors[SampleBatch.NEXT_OBS],
    )
    q_model, policy_model = policy.q_model, policy.policy_model

    gamma = policy.config["gamma"]
    # Do not bootstrap if the state is terminal
    bootstrapped = tf.squeeze(
        rewards + gamma * q_model([next_obs, policy_model(next_obs)])
    )
    q_targets = tf.compat.v1.where(dones, x=rewards, y=bootstrapped)
    policy.critic_loss = keras.losses.mean_squared_error(
        q_model([obs, actions]), q_targets
    )
    # DPG loss
    policy.actor_loss = -tf.reduce_mean(q_model([obs, policy_model(obs)]))
    return policy.actor_loss + policy.critic_loss


def get_default_config():
    """Get the default configuration for OffMAPOTFPolicy."""
    # pylint: disable=cyclic-import
    from mapo.agents.off_mapo.off_mapo import DEFAULT_CONFIG

    return DEFAULT_CONFIG


def actor_critic_gradients(policy, *_):
    """Create compute gradients ops using separate optimizers."""
    # pylint: disable=protected-access
    actor_grads = policy._actor_optimizer.get_gradients(
        policy.actor_loss, policy.policy_model.variables
    )
    critic_grads = policy._critic_optimizer.get_gradients(
        policy.critic_loss, policy.q_model.variables
    )
    # Save these for later use in build_apply_op
    policy._actor_grads_and_vars = list(zip(actor_grads, policy.policy_model.variables))
    policy._critic_grads_and_vars = list(zip(critic_grads, policy.q_model.variables))
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


def build_actor_critic_models(policy, input_dict, obs_space, action_space, config):
    """Construct actor and critic keras models, and return actor action tensor."""
    policy.q_model = build_continuous_q_function(obs_space, action_space, config)
    policy.policy_model = build_deterministic_policy(obs_space, action_space, config)

    actions = policy.policy_model(input_dict[SampleBatch.CUR_OBS])
    return actions, None


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
        `policy_delay` time(s).
        """
        should_apply_actor_opt = tf.equal(
            tf.math.mod(self.global_step, self.config["policy_delay"]), 0
        )

        def make_apply_op():
            return self._actor_optimizer.apply_gradients(self._actor_grads_and_vars)

        actor_op = tf.cond(
            should_apply_actor_opt, true_fn=make_apply_op, false_fn=tf.no_op
        )
        critic_op = self._critic_optimizer.apply_gradients(self._critic_grads_and_vars)
        # increment global step & apply ops
        with tf.control_dependencies([self.global_step.assign_add(1)]):
            return tf.group(actor_op, critic_op)


OffMAPOTFPolicy = build_tf_policy(
    name="OffMAPOTFPolicy",
    loss_fn=build_actor_critic_losses,
    get_default_config=get_default_config,
    optimizer_fn=lambda *_: None,
    gradients_fn=actor_critic_gradients,
    before_init=check_action_space,
    before_loss_init=create_separate_optimizers,
    make_action_sampler=build_actor_critic_models,
    mixins=[DelayedUpdatesMixin],
    obs_include_prev_action_reward=False,
)
