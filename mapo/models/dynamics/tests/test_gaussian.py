# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name


import gym
import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp


from mapo.models.dynamics import GaussianDynamicsModel


def get_models():
    obs_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
    models = [
        GaussianDynamicsModel(
            obs_space, action_space, layers=[], activation=tf.nn.relu
        ),
        GaussianDynamicsModel(
            obs_space, action_space, layers=[32, 16, 8], activation=tf.nn.elu
        ),
    ]
    return models


@pytest.fixture(params=get_models())
def model(request):
    return request.param


@pytest.fixture
def linear_model():
    return get_models()[0]


@pytest.fixture
def nonlinear_model():
    return get_models()[1]


@pytest.fixture
def obs_space():
    return gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)


@pytest.fixture
def action_space():
    return gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)


@pytest.fixture
def batch_size():
    return 42


@pytest.fixture
def batch_shape(batch_size):
    return (batch_size,)


@pytest.fixture
def state(obs_space, batch_size):
    return tf.stack([obs_space.sample() for _ in range(batch_size)])


@pytest.fixture
def action(action_space, batch_size):
    return tf.stack([action_space.sample() for _ in range(batch_size)])


@pytest.fixture
def inputs(obs_space, action_space):
    obs = keras.Input(shape=obs_space.shape)
    action = keras.Input(shape=action_space.shape)
    return [obs, action]


@pytest.fixture
def input_shape(obs_space, action_space):
    return (None, obs_space.shape[0] + action_space.shape[0])


@pytest.fixture
def output_shape(obs_space):
    return (None,) + obs_space.shape


def test_linear_gaussian_layers(linear_model):
    assert len(linear_model.layers) == 3
    assert not linear_model.hidden_layers
    assert isinstance(linear_model.mean_output_layer, Dense)
    assert isinstance(linear_model.log_stddev_output_layer, Dense)


def test_nonlinear_gaussian_layers(nonlinear_model):
    assert len(nonlinear_model.layers) == 6
    assert len(nonlinear_model.hidden_layers) == 3
    assert isinstance(nonlinear_model.mean_output_layer, Dense)
    assert isinstance(nonlinear_model.log_stddev_output_layer, Dense)


def test_gaussian_output_shape(model, inputs, output_shape):
    outputs = model.call(inputs)
    assert len(outputs) == 2

    mean_layer = model.mean_output_layer
    log_stddev_layer = model.log_stddev_output_layer

    assert mean_layer.output_shape == output_shape
    assert log_stddev_layer.output_shape == output_shape

    mean, log_stddev = outputs
    assert mean.shape[-1] == output_shape[-1]
    assert log_stddev.shape[-1] == output_shape[-1]


def test_linear_gaussian_call(linear_model, inputs):
    assert len(linear_model.layers) == 3
    assert not linear_model.weights
    assert not linear_model.trainable_variables

    _ = linear_model.call(inputs)
    assert len(linear_model.layers) == 3
    assert len(linear_model.weights) == 6
    assert len(linear_model.trainable_variables) == 6

    _ = linear_model.call(inputs)
    assert len(linear_model.layers) == 3
    assert len(linear_model.weights) == 6
    assert len(linear_model.trainable_variables) == 6


def test_nonlinear_gaussian_call(nonlinear_model, inputs):
    assert len(nonlinear_model.layers) == 6
    assert not nonlinear_model.weights
    assert not nonlinear_model.trainable_variables

    _ = nonlinear_model.call(inputs)
    assert len(nonlinear_model.layers) == 6
    assert len(nonlinear_model.weights) == 12
    assert len(nonlinear_model.trainable_variables) == 12

    _ = nonlinear_model.call(inputs)
    assert len(nonlinear_model.layers) == 6
    assert len(nonlinear_model.weights) == 12
    assert len(nonlinear_model.trainable_variables) == 12


def test_gaussian_distribution(model, state, action):
    dist = model.dist(state, action)
    assert isinstance(dist, tfp.distributions.Normal)
    assert dist.batch_shape == state.shape
    assert dist.dtype == state.dtype


def test_gaussian_sample(model, state, action):
    next_state = model.sample(state, action)
    assert next_state.shape == state.shape
    assert next_state.dtype == state.dtype

    sample_shape = (10,)
    next_states = model.sample(state, action, shape=sample_shape)
    assert next_states.shape == sample_shape + tuple(state.shape)
    assert next_state.dtype == state.dtype


def test_gaussian_sample_propagates_gradients(model, state, action):
    sample_shape = (10,)
    next_states = model.sample(state, action, shape=sample_shape)
    grads = tf.gradients(tf.reduce_sum(next_states), model.variables)
    assert all(grad is not None for grad in grads)


def test_gaussian_detached_sample_blocks_gradients(model, state, action):
    sample_shape = (10,)
    next_states = tf.stop_gradient(model.sample(state, action, shape=sample_shape))
    grads = tf.gradients(tf.reduce_sum(next_states), model.variables)
    assert all(grad is None for grad in grads)


def test_gaussian_log_prob(model, state, action, batch_shape):
    next_state = model.sample(state, action)
    log_prob = model.log_prob(state, action, next_state)
    assert log_prob.shape == batch_shape

    sample_shape = (10,)
    next_states = model.sample(state, action, shape=sample_shape)
    log_prob = model.log_prob(state, action, next_states)
    assert log_prob.shape == sample_shape + batch_shape


def test_gaussian_log_prob_sampled(model, state, action, batch_shape):
    next_state, log_prob = model.log_prob_sampled(state, action)
    assert next_state.shape == state.shape
    assert next_state.dtype == state.dtype
    assert log_prob.shape == batch_shape

    sample_shape = (10,)
    next_states, log_prob = model.log_prob_sampled(state, action, shape=sample_shape)
    assert next_states.shape == sample_shape + tuple(state.shape)
    assert next_state.dtype == state.dtype
    assert log_prob.shape == sample_shape + batch_shape
