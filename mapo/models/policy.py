"""Utilities for constructing policy approximators."""
from tensorflow import keras
from mapo.models.fcnet import build_fcnet
from mapo.models.layers import ActionSquashingLayer


def build_deterministic_policy(obs_space, action_space, config=None):
    """
    Contruct deterministic policy keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    """
    policy_input = keras.Input(shape=obs_space.shape)

    policy_out = build_fcnet(policy_input.shape, config=config)(policy_input)
    policy_out = keras.layers.Dense(units=action_space.shape[0])(policy_out)
    policy_out = ActionSquashingLayer(action_space)(policy_out)
    return keras.Model(inputs=policy_input, outputs=policy_out)
