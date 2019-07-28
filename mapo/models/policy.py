"""Utilities for constructing policy approximators."""
from tensorflow import keras
from mapo.models.layers import ActionSquashingLayer

DEFAULT_CONFIG = {"hidden_activation": "relu", "hidden_units": [400, 300]}


def build_deterministic_policy(obs_space, action_space, config=None):
    """
    Contruct deterministic policy keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    """
    config = config or {}
    config = {**DEFAULT_CONFIG, **config}

    policy_input = keras.Input(shape=obs_space.shape)
    activation = config["hidden_activation"]
    policy_out = policy_input
    for units in config["hidden_units"]:
        policy_out = keras.layers.Dense(units=units, activation=activation)(policy_out)

    policy_out = keras.layers.Dense(units=action_space.shape[0])(policy_out)
    policy_out = ActionSquashingLayer(action_space)(policy_out)
    return keras.Model(inputs=policy_input, outputs=policy_out)
