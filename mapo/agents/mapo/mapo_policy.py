"""MAPO Tensorflow Policy."""
import tensorflow as tf
from gym.spaces import Box

from ray.rllib.policy import build_tf_policy
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing


POLICY_SCOPE = "policy"
Q_SCOPE = "critic"


def postprocess_returns(policy, sample_batch, other_agent_batches=None, episode=None):
    # pylint: disable=unused-argument
    """Add trajectory returns."""
    return compute_advantages(sample_batch, 0.0, policy.config["gamma"], use_gae=False)


def build_actor_critic_losses(policy, batch_tensors):
    """Contruct actor (DPG) and critic (Fitted Q) tf losses."""
    # Q-values for actions & observation in input batch
    with tf.variable_scope(Q_SCOPE):
        q_values, policy.q_model = build_continuous_q_function(
            batch_tensors[SampleBatch.CUR_OBS],
            batch_tensors[SampleBatch.ACTIONS],
            policy.observation_space,
            policy.action_space,
            policy.config,
        )

    # Fitted Q loss (using trajectory returns)
    critic_loss = tf.reduce_mean(
        tf.square(q_values - batch_tensors[Postprocessing.ADVANTAGES])
    )

    # Q-values for policy actions calculated from observations in input batch
    with tf.variable_scope(Q_SCOPE, reuse=True):
        q_policy_values, _ = build_continuous_q_function(
            batch_tensors[SampleBatch.CUR_OBS],
            policy.actions,
            policy.observation_space,
            policy.action_space,
            policy.config,
        )

    # DPG loss
    actor_loss = -tf.reduce_mean(q_policy_values)

    return actor_loss + critic_loss


def check_action_space(policy, obs_space, action_space, config):
    # pylint: disable=unused-argument
    """Check if the action space is suited to DPG."""
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


def build_continuous_q_function(obs, actions, obs_space, action_space, config):
    """Construct Q function tf graph."""
    if config["use_state_preprocessor"]:
        q_model = ModelCatalog.get_model(
            {"obs": obs}, obs_space, action_space, 1, config["model"]
        )
        q_out = tf.concat([q_model.last_layer, actions], axis=1)
    else:
        q_model = None
        q_out = tf.concat([obs, actions], axis=1)

    activation = getattr(tf.nn, config["critic_hidden_activation"])
    for hidden in config["critic_hiddens"]:
        q_out = tf.layers.dense(q_out, units=hidden, activation=activation)
    q_values = tf.layers.dense(q_out, units=1, activation=None)

    return q_values, q_model


def build_deterministic_policy(obs, obs_space, action_space, config):
    """Contruct deterministic policy tf graph."""
    if config["use_state_preprocessor"]:
        model = ModelCatalog.get_model(
            {"obs": obs}, obs_space, action_space, 1, config["model"]
        )
        action_out = model.last_layer
    else:
        model = None
        action_out = obs

    activation = getattr(tf.nn, config["actor_hidden_activation"])
    for hidden in config["actor_hiddens"]:
        action_out = tf.layers.dense(action_out, units=hidden, activation=activation)
    action_out = tf.layers.dense(
        action_out, units=action_space.shape[0], activation=None
    )

    # Use sigmoid to scale to [0,1], but also double magnitude of input to
    # emulate behaviour of tanh activation used in DDPG and TD3 papers.
    sigmoid_out = tf.nn.sigmoid(2 * action_out)
    # Rescale to actual env policy scale
    # (shape of sigmoid_out is [batch_size, dim_actions], so we reshape to
    # get same dims)
    action_range = (action_space.high - action_space.low)[None]
    low_action = action_space.low[None]
    actions = action_range * sigmoid_out + low_action

    return actions, model


def build_actor_action_ops(policy, input_dict, obs_space, action_space, config):
    """Construct action sampling tf ops."""
    with tf.variable_scope(POLICY_SCOPE):
        policy.actions, policy.policy_model = build_deterministic_policy(
            input_dict[SampleBatch.CUR_OBS], obs_space, action_space, config
        )
    return policy.actions, None


def get_default_config():
    """Get the default configuration for MAPOTFPolicy."""
    from mapo.agents.mapo.mapo import DEFAULT_CONFIG  # pylint: disable=cyclic-import

    return DEFAULT_CONFIG


MAPOTFPolicy = build_tf_policy(
    name="MAPOTFPolicy",
    loss_fn=build_actor_critic_losses,
    get_default_config=get_default_config,
    postprocess_fn=postprocess_returns,
    before_init=check_action_space,
    make_action_sampler=build_actor_action_ops,
    # optimizer_fn=lambda: None,
)
