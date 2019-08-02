"""Tests for Q keras model."""
# pylint: disable=missing-docstring
import tensorflow as tf
from tensorflow import keras
from mapo.tests.mock_env import MockEnv
from mapo.models.q_function import build_continuous_q_function


def test_q_function_output_has_consistent_shape():
    env = MockEnv()
    ob_space, ac_space = env.observation_space, env.action_space
    q_model = build_continuous_q_function(ob_space, ac_space)
    obs = tf.placeholder(dtype=ob_space.dtype, shape=(None,) + ob_space.shape)
    actions = tf.placeholder(dtype=ac_space.dtype, shape=(None,) + ac_space.shape)
    output = q_model([obs, actions])
    assert output.shape.as_list() == tf.TensorShape((None, 1)).as_list()


def test_q_function_model_is_serializable():
    env = MockEnv()
    ob_space, ac_space = env.observation_space, env.action_space
    q_model = build_continuous_q_function(ob_space, ac_space)
    q_config = q_model.get_config()
    keras.Model.from_config(q_config)
