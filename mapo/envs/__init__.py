# pylint: disable=missing-docstring
import abc
import gym
import tensorflow as tf

from ray.rllib.models import ModelCatalog


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

            next_state, log_prob = self._transition_fn(
                self._state_placeholder, self._action_placeholder
            )
            self._transition_tensor = (
                tf.squeeze(next_state, axis=0),
                tf.squeeze(log_prob, axis=0),
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
    def _transition_fn(self, state, action, n_samples=1):
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
        next_state, log_prob = self._sess.run(
            self._transition_tensor, feed_dict=feed_dict
        )
        return next_state, log_prob

    def _reward(self, state, action, next_state):
        feed_dict = {
            self._state_placeholder: state,
            self._action_placeholder: action,
            self._next_state_placeholder: next_state,
        }
        return self._sess.run(self._reward_tensor, feed_dict=feed_dict)


class TimeAwareTFEnv(MAPOTFCustomEnv):
    def __init__(self, env, horizon=20):
        # pylint: disable=super-init-not-called
        self._env = env
        self._horizon = horizon

        self._state = None

        self.observation_space = gym.spaces.Tuple(
            (env.observation_space, gym.spaces.Discrete(horizon + 1))
        )
        self.action_space = env.action_space

    @property
    def unwrapped(self):
        return self._env

    @property
    def horizon(self):
        return self._horizon

    def render(self, mode="human"):
        pass

    def close(self):
        self._env.close()

    @property
    def start_state(self):
        return self._env.start_state, 0

    @property
    def obs(self):
        return self._state

    def _transition_fn(self, state, action, n_samples=1):
        # pylint: disable=protected-access
        state, time = state
        next_state, log_prob = self._env._transition_fn(state, action, n_samples)
        time = increment_one_hot_time(time, n_samples)
        return (next_state, time), log_prob

    def _transition_log_prob_fn(self, state, action, next_state):
        # pylint: disable=protected-access
        state, _ = state
        next_state, _ = next_state
        log_prob = self._env._transition_log_prob_fn(state, action, next_state)
        return log_prob

    def _reward_fn(self, state, action, next_state):
        # pylint: disable=protected-access
        state, _ = state
        next_state, _ = next_state
        reward = self._env._reward_fn(state, action, next_state)
        return reward

    def _terminal(self):
        # pylint: disable=protected-access
        return self._env._terminal() or self._state[1] >= self._horizon

    def _info(self):
        return self._env._info()  # pylint: disable=protected-access

    def _transition(self, state, action):
        # pylint: disable=protected-access
        state, time = state
        next_state, log_prob = self._env._transition(state, action)
        return (next_state, time + 1), log_prob

    def _reward(self, state, action, next_state):
        # pylint: disable=protected-access
        state, _ = state
        next_state, _ = next_state
        return self._env._reward(state, action, next_state)


def increment_one_hot_time(time, n_samples):
    time_int = tf.argmax(time, axis=-1)
    new_time_int = time_int + 1
    new_time = tf.one_hot(new_time_int, depth=time.shape[-1])
    return tf.stack([new_time] * n_samples)
