"""MAPO Tensorflow Policy."""
import tensorflow as tf
from tensorflow import keras
from gym.spaces import Box

from ray.rllib.policy import build_tf_policy
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import compute_advantages, Postprocessing

from mapo.models import QNetwork, PolicyNetwork


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

    critic_loss = keras.losses.mean_squared_error(q_function(obs, actions), returns)
    # DPG loss
    actor_loss = -tf.reduce_mean(q_function(obs, policy(obs)))

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


def build_actor_critic_models(policy, input_dict, obs_space, action_space, config):
    """Construct actor and critic keras models, and return actor action tensor."""
    policy.q_function = QNetwork(obs_space, action_space, config["critic_model"])
    policy.policy = PolicyNetwork(obs_space, action_space, config["actor_model"])

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
