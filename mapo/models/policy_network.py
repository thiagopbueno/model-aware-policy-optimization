"""Fully connected deterministic policy network for continuous action spaces."""
from tensorflow import keras

POLICY_NETWORK_DEFAULTS = {
    # Nonlinearity for fully connected net (tanh, relu)
    "fcnet_activation": "relu",
    # Number of hidden layers for fully connected net
    "fcnet_hiddens": [64, 64],
}


class PolicyNetwork(keras.Model):
    """
    Deterministic policy keras model.

    Assumes both obs_space and action_space are gym.spaces.Box instances.
    """

    def __init__(self, obs_space, action_space, config):
        super().__init__()
        self.obs_space = obs_space
        self.action_space = action_space

        policy_input = keras.Input(shape=obs_space.shape)
        activation = config["fcnet_activation"]
        policy_out = policy_input
        for hidden in config["fcnet_hiddens"]:
            policy_out = keras.layers.Dense(units=hidden, activation=activation)(
                policy_out
            )
        # Use sigmoid to scale to [0,1].
        out_shape = action_space.shape[0]
        policy_out = keras.layers.Dense(units=out_shape, activation="sigmoid")(
            policy_out
        )
        # Rescale to actual env policy scale
        # (shape of policy_out is [batch_size, dim_actions], so we reshape to
        # get same dims)
        action_range = (action_space.high - action_space.low)[None]
        low_action = action_space.low[None]
        policy_out = action_range * policy_out + low_action
        self.model = keras.Model(inputs=policy_input, outputs=policy_out)

    def call(self, obs):  # pylint: disable=arguments-differ
        return self.model(obs)
