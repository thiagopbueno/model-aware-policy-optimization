"""Utilities for constructing Q function approximators."""
from tensorflow import keras
from mapo.models import obs_input, action_input
from mapo.models.fcnet import build_fcnet


def build_continuous_q_function(obs_space, action_space, config=None):
    """
    Construct continuous Q function keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    Model is called on placeholder inputs so variables are created.
    """
    config = config or {}
    obs = obs_input(obs_space)
    actions = action_input(action_space)
    output = keras.layers.Concatenate(axis=-1)([obs, actions])
    output = build_fcnet(config)(output)
    values = keras.layers.Dense(units=1, activation=None)(output)
    return keras.Model(inputs=[obs, actions], outputs=values)
