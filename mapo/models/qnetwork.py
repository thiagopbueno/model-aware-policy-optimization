"""A simple, fully connected Q network for continuous action spaces."""
from tensorflow import keras


QNETWORK_DEFAULTS = {
    # how to combinate observation and action inputs
    "merge_obs_actions": "concatenate",
    # Nonlinearity for fully connected net (tanh, relu)
    "fcnet_activation": "relu",
    # Number of hidden layers for fully connected net
    "fcnet_hiddens": [64, 64],
}


MERGE_STRATEGIES = {
    "add": keras.layers.Add,
    "subtract": keras.layers.Subtract,
    "average": keras.layers.Average,
    "minimum": keras.layers.Minimum,
    "maximum": keras.layers.Maximum,
    "concatenate": keras.layers.Concatenate,
}


class QNetwork(keras.Model):
    """Model that approximates a Q-value function."""

    def __init__(self, obs_space, action_space, config):
        """
        Declare the Dense layers that compose the fully connected network.
        No observation preprocessing is done. Assumes both obs_space and
        action_space are gym.spaces.Box instances.
        """
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space

        obs_input = keras.Input(shape=obs_space.shape)
        action_input = keras.Input(shape=action_space.shape)
        merge_layer = MERGE_STRATEGIES[config["merge_obs_actions"]]()
        output = merge_layer([obs_input, action_input])
        activation = config["fcnet_activation"]
        for hidden in config["fcnet_hiddens"]:
            output = keras.layers.Dense(units=hidden, activation=activation)(output)
        output = keras.layers.Dense(units=1, activation=None)(output)

        self.model = keras.Model(inputs=[obs_input, action_input], outputs=output)

    def call(self, observations, actions):  # pylint: disable=arguments-differ
        """Build the computation graph, taking observations as flattened vectors."""
        return self.model([observations, actions])
