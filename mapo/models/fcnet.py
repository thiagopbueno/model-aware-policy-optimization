"""Utilities for constructing fully connected networks in Keras."""
import numpy as np
from tensorflow import keras


DEFAULT_CONFIG = {
    "layers": (32,),
    # Valid options: keras activation name or config dict
    "activation": "relu",
    # Whether to use Layer Normalization between hidden layers
    "layer_normalization": False,
    # Valid options: keras initializer name or config dict
    "kernel_initializer": {"class_name": "orthogonal", "config": {"gain": np.sqrt(2)}},
    # Weight regularization
    "kernel_regularizer": keras.regularizers.l1_l2(l1=0.01, l2=0.01),
    # Size of the output layer, if any
    "output_layer": None,
    # Valid options: keras activation name or config dict
    "output_activation": None,
    # Valid options: keras initializer name or config dict
    "output_kernel_initializer": {"class_name": "orthogonal", "config": {"gain": 0.01}},
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
    kernel_regularizer = config["kernel_regularizer"]

    model = keras.Sequential()
    for units in config["layers"]:
        if layer_norm:
            model.add(
                keras.layers.Dense(
                    units=units,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )
            model.add(keras.layers.LayerNormalization())
            model.add(keras.activations.get(activation))
        else:
            model.add(
                keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                )
            )
    if config["output_layer"]:
        model.add(
            keras.layers.Dense(
                units=config["output_layer"],
                activation=config["output_activation"],
                kernel_initializer=config["output_kernel_initializer"],
                kernel_regularizer=kernel_regularizer,
            )
        )
    return model
