"""Dummy gym.Env subclasses."""
import gym
from gym.spaces import Box
import numpy as np


_DEFAULT_CONFIG = {"action_dim": 4, "action_low": -1, "action_high": 1}


class MockEnv(gym.Env):  # pylint: disable=abstract-method
    """Dummy environment with continuous action space."""

    def __init__(self, config=None):
        config = config or {}
        self.config = {**_DEFAULT_CONFIG, **config}
        self.horizon = 200
        self.time = 0
        self.observation_space = Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        action_dim = self.config["action_dim"]
        low, high = self.config["action_low"], self.config["action_high"]
        self.action_space = Box(
            low=low, high=high, shape=(action_dim,), dtype=np.float32
        )

    def reset(self):
        self.time = 0
        return self.observation_space.sample()

    def step(self, action):
        self.time += 1
        return self.observation_space.sample(), 1, self.time >= self.horizon, {}
