"""Dummy gym.Env subclasses."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gym.spaces import Box

from mapo.envs import MAPOTFCustomEnv


_DEFAULT_CONFIG = {
    "action_dim": 4,
    "action_low": -1,
    "action_high": 1,
    "fixed_state": False,
}


class MockEnv(MAPOTFCustomEnv):  # pylint: disable=abstract-method
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
        super().__init__(self.observation_space.shape, self.action_space.shape)
        self.next_state = (
            self.observation_space.sample() if self.config["fixed_state"] else None
        )

    @property
    def start_state(self):
        return (
            self.next_state
            if self.next_state is not None
            else self.observation_space.sample()
        )

    @property
    def obs(self):
        return self._state

    def _transition_fn(self, state, action):  # pylint: disable=unused-argument
        dist = self._next_state_dist()
        next_state = dist.sample()
        log_prob = dist.log_prob(tf.stop_gradient(next_state))
        return next_state, log_prob

    def _transition_log_prob_fn(self, state, action, next_state):
        # pylint: disable=unused-argument
        dist = self._next_state_dist()
        return dist.log_prob(tf.stop_gradient(next_state))

    def _reward_fn(self, state, action, next_state):  # pylint: disable=unused-argument
        return tf.ones(tf.shape(next_state)[:-1])

    def _next_state_dist(self):
        return tfp.distributions.Uniform(
            low=self.observation_space.low, high=self.observation_space.high
        )

    def _terminal(self):
        return self.time >= self.horizon

    def _info(self):
        return {}
