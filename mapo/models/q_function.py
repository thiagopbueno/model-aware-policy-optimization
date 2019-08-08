"""Utilities for constructing Q function approximators."""
from tensorflow import keras

DEFAULT_CONFIG = {"hidden_activation": "relu", "hidden_units": [64, 64]}


def build_continuous_q_function(obs_space, action_space, config=None):
    """
    Construct continuous Q function keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    """
    config = config or {}
    config = {**DEFAULT_CONFIG, **config}

    obs_input = keras.Input(shape=obs_space.shape)
    action_input = keras.Input(shape=action_space.shape)
    activation = config["hidden_activation"]

    output = keras.layers.concatenate([obs_input, action_input])
    for hidden in config["hidden_units"]:
        output = keras.layers.Dense(units=hidden, activation=activation)(output)
    output = keras.layers.Dense(units=1, activation=None)(output)
    return keras.Model(inputs=[obs_input, action_input], outputs=output)
