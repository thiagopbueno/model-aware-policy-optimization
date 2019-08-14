# pylint: disable=missing-docstring

import numpy as np
import gym


DEFAULT_CONFIG = {
    "start": [-10.0, -10.0],
    "end": [10.0, 10.0],
    "action_lower_bound": [-1.0, -1.0],
    "action_upper_bound": [1.0, 1.0],
    "deceleration_zones": {"center": [[0.0, 0.0]], "decay": [2.0]},
    "noise": {"mean": [0.0, 0.0], "cov": [[0.3, 0.0], [0.0, 0.3]]},
    "horizon": 20,
}


class NavigationEnv(gym.Env):
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

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self._action_lower_bound, high=self._action_upper_bound
        )

        self._state = None
        self._timestep = None

    def step(self, action):
        next_state = self._transition(self._state, action)
        reward = self._reward(self._state)
        done = self._terminal()
        info = {}
        self._state = next_state
        self._timestep += 1
        return self.obs, reward, done, info

    def reset(self):
        self._state = self._start.copy()
        self._timestep = 0
        return self.obs

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    @property
    def obs(self):
        return self._state

    def _terminal(self):
        reached_goal = np.allclose(self._state, self._end, atol=1e-1)
        timeout = self._timestep > self._horizon
        return reached_goal or timeout

    def _transition(self, state, action):
        noise = self._sample_noise()

        deceleration = 1.0
        if self._deceleration_zones:
            deceleration = self._deceleration()

        next_state = state + (deceleration * action) + noise
        return next_state.astype(np.float32)

    def _reward(self, state):
        return -np.linalg.norm(state - self._end)

    def _sample_noise(self):
        mean = self._noise["mean"]
        cov = self._noise["cov"]
        sample = np.random.multivariate_normal(mean, cov)
        return sample

    def _deceleration(self):
        decay = self._deceleration_decay
        center = self._deceleration_center
        distance = self._distance_to_deceleration_zones(center)
        deceleration = np.prod(self._deceleration_factors(decay, distance))
        return deceleration

    def _distance_to_deceleration_zones(self, center):
        distance = np.linalg.norm(self._state - center, axis=1)
        return distance

    @staticmethod
    def _deceleration_factors(decay, distance):
        factor = 2 / (1.0 + np.exp(-decay * distance)) - 1.0
        return factor
