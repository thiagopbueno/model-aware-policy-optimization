# pylint: disable=missing-docstring

import numpy as np
import gym
import tensorflow as tf
import tensorflow_probability as tfp

from mapo.envs import MAPOTFCustomEnv


DEFAULT_CONFIG = {
    "start": -10.0,
    "end": 10.0,
    "action_lower_bound": -1.0,
    "action_upper_bound": 1.0,
    "noise": {"mean": 0.0, "stddev": 0.3},
    "r_max": 100.0,
    "n_random_walks": 10,
}


class Navigation1DEnv(MAPOTFCustomEnv):
    """Navigation1DEnv implements a linear Navigation in 1-D.

    Additional random-walk-based state variables are added as
    irrelevant information in the state represenation. These
    variables do not interfere with the reward function.
    """

    # pylint: disable=too-many-instance-attributes

    metadata = {"render.modes": ["human"]}

    def __init__(self, **kwargs):

        self._config = {**DEFAULT_CONFIG, **kwargs}

        self._start = np.array(self._config["start"], dtype=np.float32)
        self._end = np.array(self._config["end"], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._config["n_random_walks"] + 1,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=self._config["action_lower_bound"],
            high=self._config["action_upper_bound"],
            shape=(self._config["n_random_walks"] + 1,),
            dtype=np.float32,
        )

        self._noise = self._config["noise"]

        self._r_max = self._config["r_max"]

        super(Navigation1DEnv, self).__init__(
            state_shape=self.observation_space.shape,
            action_shape=self.action_space.shape,
        )

    def render(self, mode="human"):
        pass

    @property
    def start_state(self):
        random_walks = np.random.normal(size=self._config["n_random_walks"])
        state = np.append(self._start, random_walks)
        return state

    @property
    def obs(self):
        return self._state

    def _transition_fn(self, state, action, n_samples=1):
        position = self._next_position(state, action)
        dist, next_state = self._sample_noise(position, n_samples=n_samples)
        log_prob = dist.log_prob(next_state)
        log_prob = tf.reduce_sum(log_prob, axis=-1)
        return next_state, log_prob

    def _transition_log_prob_fn(self, state, action, next_state):
        position = self._next_position(state, action)
        dist, _ = self._sample_noise(position, n_samples=1)
        log_prob = dist.log_prob(tf.stop_gradient(next_state))
        log_prob = tf.reduce_sum(log_prob, axis=-1)
        return log_prob

    def _reward_fn(self, state, action, next_state):
        goal = tf.constant(self._end, name="goal")
        r_max = tf.constant(self._r_max, name="r_max")
        distance = tf.abs(next_state[..., 0] - goal)
        reward = r_max / (1.0 + distance)
        return reward

    def _terminal(self):
        reached_goal = np.allclose(self._state[0], self._end, atol=1e-1)
        return reached_goal

    def _info(self):
        return {}

    @staticmethod
    def _next_position(state, action):
        position = state[..., 0] + action[..., 0]
        position = tf.expand_dims(position, axis=-1)
        random_walks = state[..., 1:]
        position = tf.concat([position, random_walks], axis=-1)
        return position

    def _sample_noise(self, position, n_samples):
        tfd = tfp.distributions
        mean = position + self._noise["mean"]
        stddev = self._noise["stddev"]
        dist = tfd.Normal(loc=mean, scale=stddev)
        sample = dist.sample(sample_shape=(n_samples,))
        return dist, sample
