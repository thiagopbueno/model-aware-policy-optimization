"""Utilities for constructing Q function approximators."""
from tensorflow import keras

from mapo.models import obs_input, action_input
from mapo.models.fcnet import build_fcnet


def build_continuous_q_function(obs_space, action_space, config=None):
    """
    Construct continuous Q function keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    Model is called on placeholder inputs so that variables are created.
    """
    config = config or {}
    fc_config = dict(config, output_layer=1)
    obs = obs_input(obs_space)
    actions = action_input(action_space)
    inputs = keras.layers.Concatenate(axis=-1)([obs, actions])
    q_model = build_fcnet(fc_config)
    values = q_model(inputs)
    return keras.Model([obs, actions], values)
