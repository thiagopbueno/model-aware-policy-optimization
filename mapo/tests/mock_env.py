"""Dummy gym.Env subclasses."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gym.spaces import Box

from mapo.envs import MAPOTFCustomEnv, TimeAwareTFEnv


_DEFAULT_CONFIG = {
    "action_dim": 4,
    "action_low": -1,
    "action_high": 1,
    "fixed_state": False,
}


class MockEnv(MAPOTFCustomEnv):
    """Dummy environment implementing the MAPOTFCustomEnv interface."""

    def __init__(self, config=None):
        config = config or {}
        self.config = {**_DEFAULT_CONFIG, **config}

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

    def _transition_fn(
        self, state, action, n_samples=1
    ):  # pylint: disable=unused-argument
        dist = self._next_state_dist(state, action)
        next_state = dist.sample(sample_shape=(n_samples,))
        log_prob = tf.reduce_sum(dist.log_prob(tf.stop_gradient(next_state)), axis=-1)
        return next_state, log_prob

    def _transition_log_prob_fn(self, state, action, next_state):
        # pylint: disable=unused-argument
        dist = self._next_state_dist(state, action)
        return tf.reduce_sum(dist.log_prob(tf.stop_gradient(next_state)), axis=-1)

    def _reward_fn(self, state, action, next_state):  # pylint: disable=unused-argument
        return tf.ones(tf.shape(next_state)[:-1])

    def _next_state_dist(self, state, action):  # pylint: disable=unused-argument
        return tfp.distributions.TruncatedNormal(
            loc=tf.ones(self.observation_space.shape) * tf.norm(action, axis=-1),
            scale=1.0,
            low=self.observation_space.low,
            high=self.observation_space.high,
        )

    def _terminal(self):
        return False

    def _info(self):
        return {}

    def render(self, mode="human"):
        pass


class TimeAwareMockEnv(TimeAwareTFEnv):
    """Wrapped mock environment implementing the TimeAwareTFEnv interface."""

    def __init__(self, config=None):
        super().__init__(MockEnv(config), horizon=20)
