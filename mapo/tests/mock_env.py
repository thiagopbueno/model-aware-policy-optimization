"""Dummy gym.Env subclasses."""
import gym
from gym.spaces import Box
import numpy as np


_DEFAULT_CONFIG = {
    "action_dim": 4,
    "action_low": -1,
    "action_high": 1,
    "fixed_state": False,
}


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
        self.next_state = (
            self.observation_space.sample() if self.config["fixed_state"] else None
        )

    def reset(self):
        self.time = 0
        return self._next_state()

    def step(self, action):
        self.time += 1
        return self._next_state(), 1, self.time >= self.horizon, {}

    def _next_state(self):
        return (
            self.next_state
            if self.next_state is not None
            else self.observation_space.sample()
        )
