"""Utilities for constructing fully connected networks in Keras."""
import numpy as np
from tensorflow import keras


DEFAULT_CONFIG = {
    "layers": (400, 300),
    "activation": "relu",
    "layer_normalization": False,
    # Valid options: keras initializer name or "normc"
    "kernel_initializer": {"class_name": "orthogonal", "config": {"gain": np.sqrt(2)}},
}


def build_fcnet(config):
    """Construct a fully connected Keras model.

    Variables are lazily initialized.

    Return:
        model(keras.Sequential): a fully connected network
    """
    config = {**DEFAULT_CONFIG, **config}
    activation = config["activation"]
    layer_norm = config["layer_normalization"]
    kernel_initializer = config["kernel_initializer"]

    model = keras.Sequential()
    for units in config["layers"]:
        if layer_norm:
            model.add(
                keras.layers.Dense(units=units, kernel_initializer=kernel_initializer)
            )
            model.add(keras.layers.LayerNormalization())
            model.add(keras.activations.get(activation))
        else:
            model.add(
                keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                )
            )
    return model
