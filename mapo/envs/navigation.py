# pylint: disable=missing-docstring

import numpy as np
import gym
import tensorflow as tf
import tensorflow_probability as tfp

from mapo.envs import MAPOTFCustomEnv


DEFAULT_CONFIG = {
    "start": [-10.0, -10.0],
    "end": [10.0, 10.0],
    "action_lower_bound": [-1.0, -1.0],
    "action_upper_bound": [1.0, 1.0],
    "deceleration_zones": {"center": [[0.0, 0.0]], "decay": [2.0]},
    "noise": {"mean": [0.0, 0.0], "cov": [[0.3, 0.0], [0.0, 0.3]]},
    "horizon": 20,
}


class NavigationEnv(MAPOTFCustomEnv):
    """NavigationEnv implements a gym environment for the Navigation
    domain.

    The agent must navigate from a start position to and end position.
    Its actions represent displacements in the 2D plane. Gaussian noise
    is added to the final position as to incorporate uncertainty in the
    transition. Additionally, the effect of an action might be decreased
    by a scalar factor dependent on the proximity of deceleration zones.

    Please refer to the AAAI paper for further details:

    Bueno, T.P., de Barros, L.N., Mauá, D.D. and Sanner, S., 2019, July.
    Deep Reactive Policies for Planning in Stochastic Nonlinear Domains.
    In Proceedings of the AAAI Conference on Artificial Intelligence.
    """

    # pylint: disable=too-many-instance-attributes

    metadata = {"render.modes": ["human"]}

    def __init__(self, **kwargs):

        self._config = {**DEFAULT_CONFIG, **kwargs}

        self._start = np.array(self._config["start"], dtype=np.float32)
        self._end = np.array(self._config["end"], dtype=np.float32)

        self._action_lower_bound = np.array(
            self._config["action_lower_bound"], dtype=np.float32
        )
        self._action_upper_bound = np.array(
            self._config["action_upper_bound"], dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self._action_lower_bound, high=self._action_upper_bound
        )

        self._deceleration_zones = self._config["deceleration_zones"]
        if self._deceleration_zones:
            self._deceleration_decay = np.array(
                self._deceleration_zones["decay"], dtype=np.float32
            )
            self._deceleration_center = np.array(
                self._deceleration_zones["center"], dtype=np.float32
            )

        self._noise = self._config["noise"]

        self._horizon = self._config["horizon"]

        super(NavigationEnv, self).__init__(state_shape=(2,), action_shape=(2,))

    def render(self, mode="human"):
        pass

    @property
    def start_state(self):
        return self._start

    @property
    def obs(self):
        return self._state

    def _transition_fn(self, state, action):
        deceleration = 1.0
        if self._deceleration_zones:
            deceleration = self._deceleration()

        noise = self._sample_noise()
        next_state = state + (deceleration * action) + noise
        return next_state

    def _reward_fn(self, state, action, next_state):
        goal = tf.constant(self._end, name="goal")
        return -tf.norm(state - goal)  # pylint: disable=invalid-unary-operand-type

    def _terminal(self):
        reached_goal = np.allclose(self._state, self._end, atol=1e-1)
        timeout = self._timestep > self._horizon
        return reached_goal or timeout

    def _info(self):
        return {}

    def _sample_noise(self):
        mean = self._noise["mean"]
        cov = self._noise["cov"]
        tfd = tfp.distributions
        dist = tfd.MultivariateNormalFullCovariance(loc=mean, covariance_matrix=cov)
        sample = dist.sample()
        return sample

    def _deceleration(self):
        decay = tf.constant(self._deceleration_decay, name="decay")
        center = tf.constant(self._deceleration_center, name="center")
        distance = self._distance_to_deceleration_zones(center)
        deceleration = tf.reduce_prod(self._deceleration_factors(decay, distance))
        return deceleration

    def _distance_to_deceleration_zones(self, center):
        distance = tf.norm(self._state_placeholder - center, axis=1)
        return distance

    @staticmethod
    def _deceleration_factors(decay, distance):
        factor = 2 / (1.0 + tf.exp(-decay * distance)) - 1.0
        return factor
