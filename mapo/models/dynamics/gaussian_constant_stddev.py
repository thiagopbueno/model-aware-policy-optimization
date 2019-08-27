# pylint: disable=missing-docstring

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp

from ray.rllib.utils.annotations import override
from mapo.models.layers import TimeAwareObservationLayer
from mapo.envs import increment_one_hot_time


class GaussianConstantStdDevDynamicsModel(keras.Model):
    """GaussianDynamicsModel implements a state-action conditional
    probabilistic transition model based on an unimodal Gaussian distribution
    with parameterized mean and parameterized diagonal covariance matrix.

    The Gaussian mean and diagonal covariance matrix are both parameterized by
    feedforward networks with given 'layers' and 'activation'.

    Args:
        obs_space(gym.spaces.Box): The environment observation space.
        action_space(gym.spaces.Box): The environment action space.

    Keyword arguments:
        layers(List[int]): A list of hidden layer units.
        activation(tf.Tensor): An activation function op.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, obs_space, action_space, **kwargs):
        super().__init__()

        self.obs_space = obs_space
        self.action_space = action_space

        self.config = {**kwargs}

        self.obs_layer = TimeAwareObservationLayer(
            self.obs_space,
            obs_embedding_dim=self.config.get("obs_embedding_dim", 32),
            input_layer_norm=self.config.get("input_layer_norm", False),
            ignore_time=True,
        )

        self.hidden_layers = []
        activation = self.config.get("activation", "relu")
        kernel_initializer = self.config.get("kernel_initializer", "orthogonal")
        for units in self.config["layers"]:
            self.hidden_layers.append(
                Dense(
                    units, activation=activation, kernel_initializer=kernel_initializer
                )
            )
        output_kernel_initializer = self.config.get(
            "output_kernel_initializer",
            {"class_name": "orthogonal", "config": {"gain": 0.01}},
        )
        if hasattr(self.obs_space, "original_space"):
            logit_dim = self.obs_space.original_space.spaces[0].shape[0]
        else:
            logit_dim = self.obs_space.shape[0]
        self.mean_output_layer = Dense(
            logit_dim, kernel_initializer=output_kernel_initializer
        )
        self.log_stddev_output = self.add_variable(
            "log_stddev", shape=(logit_dim,), dtype=tf.float32, initializer="zeros"
        )

    @override(keras.models.Model)
    def call(self, inputs, training=None, mask=None):
        """Returns the 2-head feedforward network output implementing
        the Gaussian distribution's mean and log std deviation.

        Args:
            inputs(Tuple(tf.Tensor, tf.Tensor)): A pair of (state, action) tensors.

        Return:
            Tuple(tf.Tensor, tf.Tensor): A pair of tensors (mean, log_stddev).
        """
        # pylint: disable=unused-argument,attribute-defined-outside-init
        obs, actions = inputs
        obs = self.obs_layer(obs)
        inputs = keras.layers.Concatenate(axis=-1)([obs, actions])
        for layer in self.hidden_layers:
            inputs = layer(inputs)
        self.mean = self.mean_output_layer(inputs)
        self.log_stddev = self.log_stddev_output
        outputs = (self.mean, self.log_stddev)
        return outputs

    def dist(self, state, action):
        """Returns the next state distribution conditioned on the state
        and the action."""
        mean, log_stddev = self.call([state, action])
        gaussian = tfp.distributions.Normal(loc=mean, scale=tf.exp(log_stddev))
        return gaussian

    def sample(self, state, action, shape=()):
        """Returns a sample of given shape conditioned on the state
        and the action."""
        dist = self.dist(state, action)
        next_state = dist.sample(shape)
        if isinstance(state, (tuple, list)):
            _, time = state
            next_state = next_state, increment_one_hot_time(time, shape[0])
        return next_state

    def log_prob(self, state, action, next_state):
        """Returns the scalar log-probability for the transition given by
        (state, action, next_state)."""
        if isinstance(state, (tuple, list)):
            next_state, _ = next_state
        dist = self.dist(state, action)
        log_probs = dist.log_prob(next_state)
        log_prob = tf.reduce_sum(log_probs, axis=-1)
        return log_prob

    def log_prob_sampled(self, state, action, shape=()):
        dist = self.dist(state, action)
        next_state = tf.stop_gradient(dist.sample(shape))
        log_probs = dist.log_prob(next_state)
        log_prob = tf.reduce_sum(log_probs, axis=-1)
        if isinstance(state, (tuple, list)):
            _, time = state
            next_state = next_state, increment_one_hot_time(time, shape[0])
        return next_state, log_prob
