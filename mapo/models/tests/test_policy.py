"""Tests for policy keras model."""
# pylint: disable=missing-docstring
import tensorflow as tf
from mapo.tests.mock_env import MockEnv
from mapo.models.policy import build_deterministic_policy


def test_policy_output_has_consistent_shape():
    env = MockEnv()
    ob_space, ac_space = env.observation_space, env.action_space
    policy_model = build_deterministic_policy(ob_space, ac_space)
    obs = tf.placeholder(dtype=ob_space.dtype, shape=(None,) + ob_space.shape)
    output = policy_model(obs)
    assert output.shape.as_list() == tf.TensorShape((None,) + ac_space.shape).as_list()
