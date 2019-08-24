# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import gym.spaces as spaces
import tensorflow as tf
from ray.rllib.models import ModelCatalog

from mapo.models import obs_input
from mapo.models.layers import TimeAwareObservationLayer


def get_spaces():
    return [
        spaces.Box(low=-1.0, high=1.0, shape=(4,)),
        spaces.Tuple((spaces.Box(low=-1.0, high=1.0, shape=(4,)), spaces.Discrete(10))),
    ]


@pytest.fixture(params=get_spaces())
def space(request):
    prep = ModelCatalog.get_preprocessor_for_space(request.param)
    return prep.observation_space


@pytest.fixture(params=[True, False])
def input_layer_norm(request):
    return request.param


@pytest.fixture
def layer_fn(input_layer_norm):
    return lambda space, units=32: TimeAwareObservationLayer(
        observation_space=space,
        obs_embedding_dim=units,
        input_layer_norm=input_layer_norm,
    )


def obs_tensor(space):
    return obs_input(space)


def test_initialization(space, layer_fn):
    units = 10
    layer = layer_fn(space, units=units)
    layer(obs_tensor(space))

    assert layer.observation_space == space
    assert layer.obs_embedding_dim == units
    n_layers = 1
    if layer.input_layer_norm:
        n_layers += 1
    if isinstance(space, spaces.Dict):
        n_layers += 1
    assert len(layer.variables) == n_layers * 2


def test_call(space, layer_fn):
    layer = layer_fn(space)
    output = layer(obs_tensor(space))

    assert output.shape[-1] == layer.obs_embedding_dim
    assert len(output.shape) == 2
    assert all(
        grad is not None
        for grad in tf.gradients(tf.reduce_sum(output), layer.variables)
    )
