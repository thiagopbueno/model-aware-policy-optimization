# pylint: disable=redefined-outer-name, missing-docstring, protected-access
import pytest
import numpy as np

from mapo.tests.mock_env import TimeAwareMockEnv


@pytest.fixture
def env():
    env = TimeAwareMockEnv()
    env.reset()
    return env


def test_rollout_length(env):
    action = env.action_space.sample()
    done, length = False, 0
    while not done:
        _, _, done, _ = env.step(action)
        length += 1
    assert length == env.horizon


def test_reset(env):
    state, time = env.reset()
    assert time == 0
    assert state in env.unwrapped.observation_space


def test_step(env):
    base_env = env.unwrapped
    _, before_t = env.reset()
    next_state, reward, done, info = env.step(env.action_space.sample())

    assert isinstance(reward, np.float32)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    assert isinstance(next_state, tuple)
    assert len(next_state) == 2
    assert next_state[0] in base_env.observation_space
    assert next_state[1] in range(env.horizon + 1)
    assert next_state[1] == before_t + 1
