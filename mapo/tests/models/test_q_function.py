"""Tests for Q keras model."""
# pylint: disable=missing-docstring
import tensorflow as tf
from mapo.models.q_function import build_continuous_q_function


def test_q_function_output_has_consistent_shape(spaces):
    ob_space, ac_space = spaces
    q_model = build_continuous_q_function(ob_space, ac_space)
    obs = tf.placeholder(dtype=ob_space.dtype, shape=(None,) + ob_space.shape)
    actions = tf.placeholder(dtype=ac_space.dtype, shape=(None,) + ac_space.shape)
    output = q_model([obs, actions])
    assert output.shape.as_list() == tf.TensorShape((None, 1)).as_list()
