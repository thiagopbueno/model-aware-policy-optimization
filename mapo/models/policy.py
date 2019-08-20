"""Utilities for constructing policy approximators."""
from tensorflow import keras
from ray.rllib.models.tf.misc import normc_initializer

from mapo.models import obs_input
from mapo.models.fcnet import build_fcnet
from mapo.models.layers import ActionSquashingLayer


def build_deterministic_policy(obs_space, action_space, config=None):
    """
    Contruct deterministic policy keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    Model is called on placeholder inputs so variables are created.
    """
    config = config or {}
    obs = obs_input(obs_space)
    policy_out = build_fcnet(config)(obs)
    policy_out = keras.layers.Dense(
        units=action_space.shape[0], kernel_initializer=normc_initializer(0.01)
    )(policy_out)
    actions = ActionSquashingLayer(action_space)(policy_out)
    return keras.Model(inputs=obs, outputs=actions)
