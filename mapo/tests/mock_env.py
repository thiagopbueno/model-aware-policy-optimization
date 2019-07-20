"""Dummy gym.Env subclasses."""
import gym
from gym.spaces import Box
import numpy as np


class MockEnv(gym.Env):  # pylint: disable=abstract-method
    """Dummy environment with continuous action space."""

    def __init__(self, config=None):
        self.config = config
        self.horizon = 200
        self.time = 0
        self.observation_space = Box(high=1, low=-1, shape=(4,), dtype=np.float32)
        self.action_space = Box(high=1, low=-1, shape=(2,), dtype=np.float32)

    def reset(self):
        self.time = 0
        return self.observation_space.sample()

    def step(self, action):
        self.time += 1
        return self.observation_space.sample(), 1, self.time >= self.horizon, {}
