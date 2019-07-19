"""MAPO Tensorflow Policy."""
import tensorflow as tf
from tensorflow import keras
from gym.spaces import Box

from ray.rllib.policy import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing


def postprocess_returns(policy, sample_batch, other_agent_batches=None, episode=None):
    # pylint: disable=unused-argument
    """Add trajectory returns."""
    return compute_advantages(sample_batch, 0.0, policy.config["gamma"], use_gae=False)


def build_actor_critic_losses(policy, batch_tensors):
    """Contruct actor (DPG) and critic (Fitted Q) tf losses."""
    # Fitted Q loss (using trajectory returns)
    obs, actions, returns = (
        batch_tensors[SampleBatch.CUR_OBS],
        batch_tensors[SampleBatch.ACTIONS],
        batch_tensors[Postprocessing.ADVANTAGES],
    )
    q_function, policy = policy.q_function, policy.policy

    critic_loss = keras.losses.mean_squared_error(q_function([obs, actions]), returns)
    # DPG loss
    actor_loss = -tf.reduce_mean(q_function([obs, policy(obs)]))

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


def build_continuous_q_function(obs_space, action_space, config):
    """
    Construct continuous Q function keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    """
    obs_input = keras.Input(shape=obs_space.shape)
    action_input = keras.Input(shape=action_space.shape)
    activation = config["critic_hidden_activation"]

    output = keras.layers.concatenate([obs_input, action_input])
    for hidden in config["critic_hiddens"]:
        output = keras.layers.Dense(units=hidden, activation=activation)(output)
    output = keras.layers.Dense(units=1, activation=None)(output)
    return keras.Model(inputs=[obs_input, action_input], outputs=output)


def build_deterministic_policy(obs_space, action_space, config):
    """
    Contruct deterministic policy keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    """

    policy_input = keras.Input(shape=obs_space.shape)
    activation = config["actor_hidden_activation"]
    policy_out = policy_input
    for hidden in config["actor_hiddens"]:
        policy_out = keras.layers.Dense(units=hidden, activation=activation)(policy_out)

    # Use sigmoid to scale to [0,1].
    policy_out = keras.layers.Dense(units=action_space.shape[0], activation="sigmoid")(
        policy_out
    )
    # Rescale to actual env policy scale
    # (shape of policy_out is [batch_size, dim_actions], so we reshape to
    # get same dims)
    action_range = (action_space.high - action_space.low)[None]
    low_action = action_space.low[None]
    policy_out = action_range * policy_out + low_action
    return keras.Model(inputs=policy_input, outputs=policy_out)


def build_actor_critic_models(policy, input_dict, obs_space, action_space, config):
    """Construct actor and critic keras models, and return actor action tensor."""
    policy.q_function = build_continuous_q_function(obs_space, action_space, config)
    policy.policy = build_deterministic_policy(obs_space, action_space, config)

    actions = policy.policy(input_dict[SampleBatch.CUR_OBS])
    return actions, None


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
    make_action_sampler=build_actor_critic_models,
)
