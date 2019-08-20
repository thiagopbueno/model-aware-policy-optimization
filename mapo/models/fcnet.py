"""Utilities for constructing fully connected networks in Keras."""
from tensorflow import keras
from ray.rllib.models.tf.misc import normc_initializer


DEFAULT_CONFIG = {
    "layers": (400, 300),
    "activation": "relu",
    "layer_normalization": False,
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

    model = keras.Sequential()
    for units in config["layers"]:
        if layer_norm:
            model.add(
                keras.layers.Dense(
                    units=units, kernel_initializer=normc_initializer(1.0)
                )
            )
            model.add(keras.layers.LayerNormalization())
            model.add(keras.activations.get(activation))
        else:
            model.add(
                keras.layers.Dense(
                    units=units,
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0),
                )
            )
    return model
