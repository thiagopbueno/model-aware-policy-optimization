# pylint: disable=redefined-outer-name, missing-docstring, protected-access
import pytest
import numpy as np

from mapo.envs import TimeAwareTFEnv


@pytest.fixture
def env(env_creator):
    env = TimeAwareTFEnv(env_creator(), horizon=20)
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
    env.reset()
    assert env._timestep == 0


def test_step(env):
    base_env = env.unwrapped
    before_t = env._timestep
    next_state, reward, done, info = env.step(env.action_space.sample())

    assert isinstance(reward, np.float32)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

    assert env._timestep == before_t + 1
    assert isinstance(next_state, tuple)
    assert len(next_state) == 2
    assert next_state[0] in base_env.observation_space
    assert next_state[1] in range(env.horizon + 1)
