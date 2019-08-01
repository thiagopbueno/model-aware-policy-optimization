"""Utilities for constructing fully connected networks in Keras."""
from tensorflow import keras

DEFAULT_CONFIG = {
    "hidden_activation": "relu",
    "hidden_units": [400, 300],
    "layer_normalization": False,
}


def build_fcnet(input_shape, config=None):
    """Construct a fully connected Keras model on inputs."""
    config = config or {}
    config = {**DEFAULT_CONFIG, **config}
    activation = config["hidden_activation"]
    layer_norm = config["layer_normalization"]

    inputs = output = keras.Input(shape=input_shape)
    for units in config["hidden_units"]:
        if layer_norm:
            output = keras.layers.Dense(units=units)(output)
            output = keras.layers.LayerNormalization(output)
            output = keras.activations.get(activation)(output)
        else:
            output = keras.layers.Dense(units=units, activation=activation)(output)
    return keras.Model(inputs=inputs, outputs=output)
