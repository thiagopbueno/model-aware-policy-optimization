"""Custom layer for box constrained actions."""
import tensorflow as tf
from tensorflow import keras
from ray.rllib.utils.annotations import override


class ActionSquashingLayer(keras.layers.Layer):
    """
    Squashes inputs to action space using the sigmoid function.

    Args:
        action_space (gym.spaces.Space): An action space of a gym environment
    """

    def __init__(self, action_space, **kwargs):
        super().__init__(**kwargs)
        self.action_space = action_space
        self.low = action_space.low[None]
        self.action_range = (action_space.high - action_space.low)[None]

    @override(keras.layers.Layer)
    def call(self, inputs, **kwargs):
        """Returns the tensor of the multi-head layer's output.

        Args:
            inputs (tf.Tensor): A hidden layer's output.
        Returns:
            (tf.Tensor): An action tensor.
        """
        action_range = self.action_range
        return tf.math.sigmoid(inputs / action_range) * action_range + self.low

    @override(keras.layers.Layer)
    def get_config(self):
        config = {"action_space": self.action_space}
        base_config = super().get_config()
        return {**base_config, **config}
