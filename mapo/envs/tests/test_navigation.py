# pylint: disable=missing-docstring
# pylint: disable=protected-access

import itertools
import pytest

import numpy as np
import gym
import tensorflow as tf
import tensorflow_probability as tfp

import mapo
from mapo.envs.registry import _import_navigation_v0, _import_navigation_v1
from mapo.envs.navigation import NavigationEnv, DEFAULT_CONFIG as config


mapo.register_all_environments()


def get_gym_envs():
    env0 = _import_navigation_v0(None)
    env1 = _import_navigation_v1(None)
    envs = [env0, env1]
    for env in envs:
        env.reset()
    return envs


def sample_env_states(env, batch_size):
    state = np.stack([env.observation_space.sample() for _ in range(batch_size)])
    return state


def sample_env_actions(env, batch_size):
    action = np.stack([env.action_space.sample() for _ in range(batch_size)])
    return action


@pytest.mark.parametrize("env", get_gym_envs())
def test_navigation_config(env):
    assert isinstance(env, NavigationEnv)
    assert np.allclose(env._start, config["start"])
    assert np.allclose(env._end, config["end"])
    assert np.allclose(env.action_space.low, config["action_lower_bound"])
    assert np.allclose(env.action_space.high, config["action_upper_bound"])
    assert env._noise["mean"] == config["noise"]["mean"]
    assert env._noise["cov"] == config["noise"]["cov"]


@pytest.mark.parametrize("env", get_gym_envs())
def test_observation_space(env):
    observation_space = env.observation_space
    assert isinstance(observation_space, gym.spaces.Box)
    assert not observation_space.bounded_below.all()
    assert not observation_space.bounded_above.all()
    assert observation_space.shape == (2,)
    assert observation_space.dtype == np.float32


@pytest.mark.parametrize("env", get_gym_envs())
def test_action_space(env):
    action_space = env.action_space
    assert isinstance(action_space, gym.spaces.Box)
    assert action_space.is_bounded()
    assert action_space.shape == (2,)
    assert action_space.dtype == np.float32


@pytest.mark.parametrize("env", get_gym_envs())
def test_reset(env):
    start_state = env.reset()
    assert np.allclose(start_state, env._start)
    assert np.allclose(start_state, env.obs)
    assert env._timestep == 0


@pytest.mark.parametrize("env", get_gym_envs())
def test_sample_noise(env):
    with env._graph.as_default():
        batch_size = 16
        state = tf.constant(sample_env_states(env, batch_size=batch_size))
        action = tf.constant(sample_env_actions(env, batch_size=batch_size))
        position = state + action
        dist, next_position = env._sample_noise(position)

        tfd = tfp.distributions
        assert isinstance(dist, tfd.MultivariateNormalFullCovariance)
        assert next_position.shape == (batch_size,) + env._start.shape

    with tf.Session(graph=env._graph) as sess:
        position_ = sess.run(position) + config["noise"]["mean"]
        assert np.allclose(sess.run(dist.mean()), position_)
        assert np.allclose(sess.run(dist.covariance()), config["noise"]["cov"])


@pytest.mark.parametrize("env", get_gym_envs())
def test_distance_to_deceleration_zones(env):
    with env._graph.as_default():
        if env._deceleration_zones:
            center = env._deceleration_center
            distance = env._distance_to_deceleration_zones(center)
            assert distance.shape == (len(env._deceleration_zones["center"]),)
            assert distance.shape == (len(env._deceleration_zones["decay"]),)


@pytest.mark.parametrize("env", get_gym_envs())
def test_deceleration_factors(env):
    with env._graph.as_default():
        if env._deceleration_zones:
            decay = env._deceleration_decay
            center = env._deceleration_center
            distance = env._distance_to_deceleration_zones(center)
            factor = env._deceleration_factors(decay, distance)
            assert factor.shape == (len(env._deceleration_zones["center"]),)
            assert factor.shape == (len(env._deceleration_zones["decay"]),)


@pytest.mark.parametrize("env", get_gym_envs())
def test_transition(env):
    with env._graph.as_default():
        action_low = env.action_space.low
        action_high = env.action_space.high

        state = env.obs
        action = np.random.uniform(low=action_low, high=action_high)
        next_states = []

        for _ in range(1000):
            next_state, log_prob = env._transition(state, action)
            next_states.append(next_state)
            assert next_state.shape == state.shape
            assert log_prob.shape == tuple()

        next_states = np.array(next_states, dtype=np.float32)

        if env._deceleration_zones:
            with tf.Session(graph=env._graph) as sess:
                feed_dict = {env._state_placeholder: state}
                deceleration = sess.run(env._deceleration(), feed_dict=feed_dict)
            noises = next_states - state - deceleration * action
            noise_mean = np.mean(noises, axis=0)
            noise_cov = np.cov(noises.T)
            assert np.allclose(noise_mean, env._noise["mean"], atol=1e-1)
            assert np.allclose(noise_cov, env._noise["cov"], atol=1e-1)


@pytest.mark.parametrize("env", get_gym_envs())
def test_reward(env):
    with env._graph.as_default():
        dummy_action = env.action_space.sample()
        dummy_next_state = env.observation_space.sample()

        reward = env._reward(env._end, dummy_action, dummy_next_state)
        assert np.allclose(reward, 0.0)
        reward = env._reward(env._start, dummy_action, dummy_next_state)
        assert np.allclose(reward, -np.sqrt(2 * 20 ** 2))


@pytest.mark.parametrize("env", get_gym_envs())
def test_terminal(env):
    timesteps = list(range(0, env._horizon + 1))
    states = env._end + np.random.uniform(low=1e-1, high=100.0, size=(100, 2))
    for timestep, state in itertools.product(timesteps, states):
        env._timestep = timestep
        env._state = state
        assert not env._terminal()

    for noise in np.random.normal(loc=0.0, scale=0.01, size=(100, 2)):
        env._state = env._end + noise
        assert env._terminal()

    env.reset()
    env._timestep = env._horizon + 1
    assert env._terminal()


@pytest.mark.parametrize("env", get_gym_envs())
def test_step(env):
    action_low = env.action_space.low
    action_high = env.action_space.high
    action = np.random.uniform(low=action_low, high=action_high)
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == env._start.shape
    assert obs.dtype == env._start.dtype
    assert isinstance(reward, np.float32)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


@pytest.mark.parametrize("env", get_gym_envs())
def test_transition_fn(env):
    with env._graph.as_default():
        batch_size = 32
        state_ = sample_env_states(env, batch_size)
        action_ = sample_env_actions(env, batch_size)
        state = tf.constant(state_)
        action = tf.constant(action_)
        next_state, log_prob = env._transition_fn(state, action)
        assert state.shape == next_state.shape
        assert state.dtype == next_state.dtype
        assert log_prob.shape == (batch_size,)
        assert log_prob.dtype == next_state.dtype


@pytest.mark.parametrize("env", get_gym_envs())
def test_transition_log_prob_fn(env):
    with env._graph.as_default():
        batch_size = 16
        state_ = sample_env_states(env, batch_size=batch_size)
        state = tf.constant(state_)
        action_ = sample_env_actions(env, batch_size=batch_size)
        action = tf.constant(action_)

        next_state, log_prob = env._transition_fn(state, action)
        log_prob_ = env._transition_log_prob_fn(state, action, next_state)
        assert log_prob_.shape == log_prob.shape

    with tf.Session(graph=env._graph) as sess:
        feed_dict = {
            env._state_placeholder: state_[0],
            env._action_placeholder: action_[0],
            env._next_state_placeholder: state_[0],
        }
        expected, actual = sess.run([log_prob, log_prob_], feed_dict=feed_dict)
        assert np.allclose(expected, actual)


@pytest.mark.parametrize("env", get_gym_envs())
def test_reward_fn(env):
    with env._graph.as_default():
        batch_size = 16
        state_ = sample_env_states(env, batch_size=batch_size)
        state = tf.constant(state_)
        action_ = sample_env_actions(env, batch_size=batch_size)
        action = tf.constant(action_)

        next_state, _ = env._transition_fn(state, action)
        reward = env._reward_fn(state, action, next_state)
        assert reward.shape == (batch_size,)
        assert reward.dtype == state.dtype
