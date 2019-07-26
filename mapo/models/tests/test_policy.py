"""Tests for policy keras model."""
import tensorflow as tf
from tensorflow import keras
from mapo.tests.mock_env import MockEnv
from mapo.models.policy import build_deterministic_policy
from mapo.models.layers import ActionSquashingLayer


def test_shape():
    """Check if policy output has a consistent shape."""
    env = MockEnv()
    ob_space, ac_space = env.observation_space, env.action_space
    policy_model = build_deterministic_policy(ob_space, ac_space)
    obs = tf.placeholder(dtype=ob_space.dtype, shape=(None,) + ob_space.shape)
    output = policy_model(obs)
    assert output.shape.as_list() == tf.TensorShape((None,) + ac_space.shape).as_list()


def test_is_serializable():
    """Check if policy can export config and be recreated from it."""
    env = MockEnv()
    ob_space, ac_space = env.observation_space, env.action_space
    policy_model = build_deterministic_policy(ob_space, ac_space)
    policy_config = policy_model.get_config()
    keras.Model.from_config(
        policy_config,
        custom_objects={ActionSquashingLayer.__name__: ActionSquashingLayer},
    )
