"""Utilities for constructing policy approximators."""
from tensorflow import keras

from mapo.models import obs_input
from mapo.models.fcnet import build_fcnet
from mapo.models.layers import ActionSquashingLayer, TimeAwareObservationLayer


def build_deterministic_policy(
    obs_space, action_space, obs_embedding_dim=32, input_layer_norm=False, **kwargs
):
    """
    Contruct deterministic policy keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    Model is called on placeholder inputs so that variables are created.
    """
    obs = obs_input(obs_space)

    embedding_layer = TimeAwareObservationLayer(
        obs_space,
        obs_embedding_dim=obs_embedding_dim,
        input_layer_norm=input_layer_norm,
    )
    fc_config = dict(kwargs, output_layer=action_space.shape[0])
    fc_model = build_fcnet(fc_config)
    action_layer = ActionSquashingLayer(action_space)

    output = embedding_layer(obs)
    output = fc_model(output)
    output = action_layer(output)
    return keras.Model(obs, output)
