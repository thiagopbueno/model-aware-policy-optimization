"""Model construction utilies."""

from tensorflow import keras


def obs_input(obs_space):
    """Create a keras.Input suitable to receive observations from the obs_space."""
    return keras.Input(shape=obs_space.shape)


def action_input(action_space):
    """Create a keras.Input suitable to receive actions from the action_space."""
    return keras.Input(shape=action_space.shape)
