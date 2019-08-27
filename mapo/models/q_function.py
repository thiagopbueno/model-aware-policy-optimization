"""Utilities for constructing Q function approximators."""
from tensorflow import keras

from mapo.models import obs_input, action_input
from mapo.models.fcnet import build_fcnet
from mapo.models.layers import TimeAwareObservationLayer, TimeConcatObservationLayer


def build_continuous_q_function(
    obs_space, action_space, obs_embedding_dim=32, input_layer_norm=False, **kwargs
):
    """
    Construct continuous Q function keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    Model is called on placeholder inputs so that variables are created.
    """
    obs = obs_input(obs_space)
    actions = action_input(action_space)

    # embedding_layer = TimeAwareObservationLayer(
    #     obs_space,
    #     obs_embedding_dim=obs_embedding_dim,
    #     input_layer_norm=input_layer_norm,
    # )
    embedding_layer = TimeConcatObservationLayer(obs_space)
    fc_config = dict(kwargs, output_layer=1)
    fc_model = build_fcnet(fc_config)

    output = embedding_layer(obs)
    output = keras.layers.Concatenate(axis=-1)([output, actions])
    output = fc_model(output)
    return keras.Model([obs, actions], output)
