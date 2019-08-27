"""Custom layer for dict observations with a 'time' subspace."""
import tensorflow as tf
from tensorflow import keras
from ray.rllib.utils.annotations import override


class TimeConcatObservationLayer(keras.layers.Layer):
    """
    Separate state inputs from time information.

    Args:
       units (int): Dimension of the space to project the observation into
       observation_space (gym.spaces.Space): An observation space of a gym environment
       input_layer_norm (bool): Whether to apply layer normalization
    """

    def __init__(self, observation_space, ignore_time=False, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = observation_space
        self.ignore_time = ignore_time

    @override(keras.layers.Layer)
    def call(self, inputs, **kwargs):
        """Compute a state embedding combining state and time inputs, if applicable.

        Args:
            inputs (tf.Tensor or dict): Observation tensors returned from
                ray.rllib.models.model.restore_original_dimensions
        """
        if isinstance(inputs, (tuple, list)):
            state_input, time_input = inputs
        else:
            state_input, time_input = inputs, None

        if time_input is not None and not self.ignore_time:
            time_int = tf.argmax(time_input, axis=-1)[..., None]
            time_scaled = tf.cast(time_int, tf.float32) / tf.cast(
                tf.shape(time_input)[-1], tf.float32
            )
            state_input = keras.layers.Concatenate(axis=-1)([state_input, time_scaled])

        return state_input

    @override(keras.layers.Layer)
    def get_config(self):
        config = {
            "observation_space": self.observation_space,
            "ignore_time": self.ignore_time,
        }
        base_config = super().get_config()
        return {**base_config, **config}
