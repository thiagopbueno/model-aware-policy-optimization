"""Custom layer for box constrained actions."""
import tensorflow as tf
from tensorflow import keras


class ActionSquashingLayer(keras.layers.Layer):
    """
    Squashes inputs to action space using the sigmoid function.

    Args:
        action_space (gym.spaces.Space): An action space of a gym environment
    """

    def __init__(self, action_space):
        super().__init__()
        self.low = action_space.low[None]
        self.action_range = (action_space.high - action_space.low)[None]

    def call(self, inputs, **kwargs):
        """Returns the tensor of the multi-head layer's output.

        Args:
            inputs (tf.Tensor): A hidden layer's output.
        Returns:
            (tf.Tensor): An action tensor.
        """
        action_range = self.action_range
        return tf.math.sigmoid(inputs / action_range) * action_range + self.low
