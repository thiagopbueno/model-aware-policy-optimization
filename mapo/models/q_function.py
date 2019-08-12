"""Utilities for constructing Q function approximators."""
from tensorflow import keras
from mapo.models.fcnet import build_fcnet


def build_continuous_q_function(obs_space, action_space, config=None):
    """
    Construct continuous Q function keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    """
    obs_input = keras.Input(shape=obs_space.shape)
    action_input = keras.Input(shape=action_space.shape)
    output = keras.layers.Concatenate(axis=-1)([obs_input, action_input])

    output = build_fcnet(output.shape, config=config)(output)
    output = keras.layers.Dense(units=1, activation=None)(output)
    return keras.Model(inputs=[obs_input, action_input], outputs=output)
