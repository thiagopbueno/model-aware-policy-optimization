"""Custom layer for dict observations with a 'time' subspace."""
import gym.spaces as spaces
import tensorflow as tf
from tensorflow import keras
from ray.rllib.utils.annotations import override


class TimeAwareObservationLayer(keras.layers.Layer):
    """
    Separate state inputs from time information.

    Args:
       units (int): Dimension of the space to project the observation into
       observation_space (gym.spaces.Space): An observation space of a gym environment
       input_layer_norm (bool): Whether to apply layer normalization
    """

    def __init__(
        self,
        observation_space,
        obs_embedding_dim=32,
        input_layer_norm=False,
        ignore_time=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.observation_space = observation_space
        self.obs_embedding_dim = obs_embedding_dim
        self.input_layer_norm = input_layer_norm
        self.ignore_time = ignore_time
        self.time_layer = None

        self.obs_shape = self.observation_space.shape
        original_space = getattr(self.observation_space, "original_space", None)
        if original_space is not None and isinstance(original_space, spaces.Tuple):
            self.obs_shape = original_space.spaces[0].shape
            if not self.ignore_time:
                self.time_layer = keras.layers.Dense(
                    self.obs_embedding_dim, activation="tanh", name="time_embedding"
                )

        self.state_layers = []
        if self.input_layer_norm:
            self.state_layers.append(keras.layers.LayerNormalization(name="obs_norm"))
        self.state_layers.append(
            keras.layers.Dense(units=self.obs_embedding_dim, name="obs_embedding")
        )

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

        batch_shape = tf.shape(state_input)[:-1]
        state_output = tf.reshape(state_input, (-1,) + self.obs_shape)
        for layer in self.state_layers:
            state_output = layer(state_output)
        state_output = tf.reshape(
            state_output,
            tf.concat([batch_shape, tf.constant([self.obs_embedding_dim])], axis=0),
        )

        if time_input is not None and self.time_layer is not None:
            time_output = self.time_layer(time_input)
            state_output = time_output * state_output

        return state_output

    @override(keras.layers.Layer)
    def get_config(self):
        config = {
            "observation_space": self.observation_space,
            "obs_embedding_dim": self.obs_embedding_dim,
            "input_layer_norm": self.input_layer_norm,
            "ignore_time": self.ignore_time,
        }
        base_config = super().get_config()
        return {**base_config, **config}
