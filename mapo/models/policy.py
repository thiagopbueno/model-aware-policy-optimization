"""Utilities for constructing policy approximators."""
from mapo.models import obs_input
from mapo.models.fcnet import build_fcnet
from mapo.models.layers import ActionSquashingLayer


def build_deterministic_policy(obs_space, action_space, config=None):
    """
    Contruct deterministic policy keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    Model is called on placeholder inputs so that variables are created.
    """
    config = config or {}
    fc_config = dict(config, output_layer=action_space.shape[0])
    obs = obs_input(obs_space)
    policy_model = build_fcnet(fc_config)
    policy_model.add(ActionSquashingLayer(action_space))
    policy_model(obs)
    return policy_model
