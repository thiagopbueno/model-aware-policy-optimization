# pylint: disable=missing-docstring
import abc
import gym
import tensorflow as tf

# from mapo.envs.navigation import NavigationEnv


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
        self._timestep = None

        self._graph = tf.Graph()
        with self._graph.as_default():  # pylint: disable=not-context-manager
            self._state_placeholder = tf.placeholder(tf.float32, shape=self.state_shape)
            self._action_placeholder = tf.placeholder(
                tf.float32, shape=self.action_shape
            )
            self._next_state_placeholder = tf.placeholder(
                tf.float32, shape=self.state_shape
            )

        self._sess = tf.Session(graph=self._graph)

    def step(self, action):
        next_state = self._transition(self._state, action)
        reward = self._reward(self._state, action, next_state)
        done = self._terminal()
        info = self._info()
        self._state = next_state
        self._timestep += 1
        return self.obs, reward, done, info

    def reset(self):
        self._state = self.start_state
        self._timestep = 0
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
    def _reward_fn(self, state, action, next_state):
        raise NotImplementedError

    @abc.abstractmethod
    def _terminal(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _info(self):
        raise NotImplementedError

    def _transition(self, state, action):
        # pylint: disable=attribute-defined-outside-init
        if not hasattr(self, "_transition_tensor"):
            with self._graph.as_default():  # pylint: disable=not-context-manager
                self._transition_tensor = self._transition_fn(
                    self._state_placeholder, self._action_placeholder
                )

        feed_dict = {self._state_placeholder: state, self._action_placeholder: action}
        return self._sess.run(self._transition_tensor, feed_dict=feed_dict)

    def _reward(self, state, action, next_state):
        # pylint: disable=attribute-defined-outside-init
        if not hasattr(self, "_reward_tensor"):
            with self._graph.as_default():  # pylint: disable=not-context-manager
                self._reward_tensor = self._reward_fn(
                    self._state_placeholder,
                    self._action_placeholder,
                    self._next_state_placeholder,
                )

        feed_dict = {
            self._state_placeholder: state,
            self._action_placeholder: action,
            self._next_state_placeholder: next_state,
        }
        return self._sess.run(self._reward_tensor, feed_dict=feed_dict)
