# pylint: disable=all
import tensorflow as tf
from tensorflow import keras


def main():
    # Imagine this is the shape of our batch of states (already separated from the time)
    r2tensor = tf.ones((10, 4))
    # This would be the correponding shape of next states if we sampled one for each
    r3tensor = tf.ones((1, 10, 4))

    norm = keras.layers.LayerNormalization()
    # On initialization, layer norm is called on the first batch of states
    norm(r2tensor)
    # During loss calculation, it is called on next states to calculate their values
    norm(r3tensor)


if __name__ == "__main__":
    main()
