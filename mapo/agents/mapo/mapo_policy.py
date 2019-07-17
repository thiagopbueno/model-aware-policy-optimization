"""MAPO Tensorflow Policy."""
import tensorflow as tf
from tensorflow import keras
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
    with tf.compat.v1.variable_scope(Q_SCOPE):
        policy.q_values, policy.q_preprocessor = build_continuous_q_function(
            policy.get_obs_input_dict(),
            policy.observation_space,
            policy.action_space,
            policy.config,
        )

    # Fitted Q loss (using trajectory returns)
    critic_loss = tf.reduce_mean(
        tf.square(
            policy.q_values(batch_tensors[SampleBatch.ACTIONS])
            - batch_tensors[Postprocessing.ADVANTAGES]
        )
    )

    # DPG loss
    actor_loss = -tf.reduce_mean(policy.q_values(policy.actions))

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


def build_continuous_q_function(input_dict, obs_space, action_space, config):
    """Construct the state preprocessor and critic layers."""
    if config["use_state_preprocessor"]:
        state_preprocessor = ModelCatalog.get_model(
            input_dict, obs_space, action_space, 1, config["model"]
        )
        processed_obs = state_preprocessor.last_layer
    else:
        state_preprocessor = None
        processed_obs = input_dict[SampleBatch.CUR_OBS]

    activation = config["critic_hidden_activation"]
    hiddens = [
        keras.layers.Dense(units=hidden, activation=activation)
        for hidden in config["critic_hiddens"]
    ]
    layers = hiddens + [keras.layers.Dense(units=1, activation=None)]
    model = keras.Sequential(layers=layers)

    def evaluate_actions(actions):
        return model(tf.concat([processed_obs, actions], axis=1))

    return evaluate_actions, state_preprocessor


def build_deterministic_policy(input_dict, obs_space, action_space, config):
    """Contruct deterministic policy tf graph."""
    if config["use_state_preprocessor"]:
        state_preprocessor = ModelCatalog.get_model(
            input_dict, obs_space, action_space, 1, config["model"]
        )
        action_out = state_preprocessor.last_layer
    else:
        state_preprocessor = None
        action_out = input_dict[SampleBatch.CUR_OBS]

    activation = config["actor_hidden_activation"]
    for hidden in config["actor_hiddens"]:
        action_out = keras.layers.Dense(units=hidden, activation=activation)(action_out)
    action_out = keras.layers.Dense(units=action_space.shape[0], activation=None)(
        action_out
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

    return actions, state_preprocessor


def build_actor_action_ops(policy, input_dict, obs_space, action_space, config):
    """Construct action sampling tf ops."""
    with tf.compat.v1.variable_scope(POLICY_SCOPE):
        policy.actions, policy.policy_preprocessor = build_deterministic_policy(
            input_dict, obs_space, action_space, config
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
