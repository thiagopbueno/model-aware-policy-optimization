# pylint: disable=missing-docstring
import abc
import gym
import tensorflow as tf

from ray.rllib.models import ModelCatalog
from ray.rllib.models.model import restore_original_dimensions


class MAPOTFCustomEnv(gym.Env):
    """MAPOCustomEnv defines an API on top of gym.Env in order
    to initialize the necessary TensorFlow boilerplate in
     _transition and _reward functions.

    Args:
        state_shape(Sequence[int]): The state shape.
        action_shape(Sequence[int]): The action shape.
    """

    # pylint: disable=too-many-instance-attributes
    __metaclass__ = abc.ABCMeta

    def __init__(self, state_shape, action_shape):
        self.state_shape = state_shape
        self.action_shape = action_shape

        self._state = None

        self._graph = tf.Graph()
        self._build_ops()
        self._sess = tf.Session(graph=self._graph)

    def _build_ops(self):

        with self._graph.as_default():  # pylint: disable=not-context-manager

            self._state_placeholder = tf.compat.v1.placeholder(
                tf.float32, shape=self.state_shape
            )
            self._action_placeholder = tf.compat.v1.placeholder(
                tf.float32, shape=self.action_shape
            )
            self._next_state_placeholder = tf.compat.v1.placeholder(
                tf.float32, shape=self.state_shape
            )

            self._transition_tensor = self._transition_fn(
                self._state_placeholder, self._action_placeholder
            )
            self._reward_tensor = self._reward_fn(
                self._state_placeholder,
                self._action_placeholder,
                self._next_state_placeholder,
            )

    def step(self, action):
        next_state, _ = self._transition(self._state, action)
        reward = self._reward(self._state, action, next_state)
        self._state = next_state
        done = self._terminal()
        info = self._info()
        return self.obs, reward, done, info

    def reset(self):
        self._state = self.start_state
        return self.obs

    def close(self):
        self._sess.close()

    @abc.abstractproperty
    def start_state(self):
        raise NotImplementedError

    @abc.abstractproperty
    def obs(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _transition_fn(self, state, action):
        raise NotImplementedError

    @abc.abstractmethod
    def _transition_log_prob_fn(self, state, action, next_state):
        raise NotImplementedError

    @abc.abstractmethod
    def _reward_fn(self, state, action, next_state):
        raise NotImplementedError

    @abc.abstractmethod
    def _terminal(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _info(self):
        raise NotImplementedError

    def _transition(self, state, action):
        feed_dict = {self._state_placeholder: state, self._action_placeholder: action}
        return self._sess.run(self._transition_tensor, feed_dict=feed_dict)

    def _reward(self, state, action, next_state):
        feed_dict = {
            self._state_placeholder: state,
            self._action_placeholder: action,
            self._next_state_placeholder: next_state,
        }
        return self._sess.run(self._reward_tensor, feed_dict=feed_dict)


class TimeAwareTFEnv(MAPOTFCustomEnv):
    # pylint: disable=abstract-method,
    STATE = "state"
    TIMESTEP = "timestep"

    def __init__(self, env, horizon=20):
        # pylint: disable=super-init-not-called
        self._env = env
        self._horizon = horizon
        self._timestep = None
        self.observation_space = gym.spaces.Dict(
            {
                self.STATE: env.observation_space,
                self.TIMESTEP: gym.spaces.Discrete(horizon),
            }
        )
        self._prep = ModelCatalog.get_preprocessor_for_space(self.observation_space)
        self.action_space = env.action_space

    @property
    def unwrapped(self):
        return self._env

    @property
    def horizon(self):
        return self._horizon

    def step(self, action):
        next_obs, reward, done, info = self._env.step(action)
        self._timestep += 1
        next_state = {self.STATE: next_obs, self.TIMESTEP: self._timestep}
        done = done or self._timestep >= self._horizon
        return next_state, reward, done, info

    def reset(self):
        obs = self._env.reset()
        self._timestep = 0
        state = {self.STATE: obs, self.TIMESTEP: self._timestep}
        return state

    def close(self):
        self._env.close()

    def _transition_fn(self, state, action):
        # pylint: disable=protected-access
        state, time = restore_state_tensor(state, self._prep.observation_space)
        next_state, log_prob = self._env._transition_fn(state, action)
        time = time + 1
        return {self.STATE: next_state, self.TIMESTEP: time}, log_prob

    def _transition_log_prob_fn(self, state, action, next_state):
        # pylint: disable=protected-access
        state, _ = restore_state_tensor(state, self._prep.observation_space)
        next_state, _ = restore_state_tensor(next_state, self._prep.observation_space)
        log_prob = self._env._transition_log_prob_fn(state, action, next_state)
        return log_prob

    def _reward_fn(self, state, action, next_state):
        # pylint: disable=protected-access
        state, _ = restore_state_tensor(state, self._prep.observation_space)
        next_state, _ = restore_state_tensor(next_state, self._prep.observation_space)
        reward = self._env._reward_fn(state, action, next_state)
        return reward


def restore_state_tensor(inputs, observation_space):
    batch_shape = tf.shape(inputs)[:-1]
    inputs = tf.reshape(inputs, shape=(-1, observation_space.shape[0]))
    inputs = restore_original_dimensions(
        inputs, observation_space, tensorlib=tf
    )
    if isinstance(inputs, dict):
        for key, value in inputs.items():
            inputs[key] = tf.reshape(value, shape=tf.concat([batch_shape,
                                                            tf.shape(value)[-1:]],
                                                            axis=0))

        return (
            inputs[TimeAwareTFEnv.STATE],
            inputs[TimeAwareTFEnv.TIMESTEP],
        )
    else:
        inputs = tf.reshape(inputs, shape=tf.concat([batch_shape,
                                                    tf.shape(inputs)[-1:]],
                                                    axis=0))
        return inputs, None
