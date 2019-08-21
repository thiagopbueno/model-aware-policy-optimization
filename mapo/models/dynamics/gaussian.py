# pylint: disable=missing-docstring

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp

from ray.rllib.utils.annotations import override


class GaussianDynamicsModel(keras.Model):
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

    def __init__(self, obs_space, action_space, **kwargs):
        super(GaussianDynamicsModel, self).__init__()

        self.obs_space = obs_space
        self.action_space = action_space

        self.config = {**kwargs}

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
        self.mean_output_layer = Dense(
            self.obs_space.shape[0], kernel_initializer=output_kernel_initializer
        )
        self.log_stddev_output_layer = Dense(
            self.obs_space.shape[0], kernel_initializer=output_kernel_initializer
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
        inputs = keras.layers.concatenate(inputs)
        for layer in self.hidden_layers:
            inputs = layer(inputs)
        outputs = (self.mean_output_layer(inputs), self.log_stddev_output_layer(inputs))
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
        return dist.sample(shape)

    def log_prob(self, state, action, next_state):
        """Returns the scalar log-probability for the transition given by
        (state, action, next_state)."""
        dist = self.dist(state, action)
        log_probs = dist.log_prob(next_state)
        log_prob = tf.reduce_sum(log_probs, axis=-1)
        return log_prob

    def log_prob_sampled(self, state, action, shape=()):
        dist = self.dist(state, action)
        next_state = tf.stop_gradient(dist.sample(shape))
        log_probs = dist.log_prob(next_state)
        log_prob = tf.reduce_sum(log_probs, axis=-1)
        return next_state, log_prob
