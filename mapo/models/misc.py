"""Model construction utilies."""
from tensorflow import keras
from ray.rllib.models.model import restore_original_dimensions


def obs_input(obs_space):
    """Creates placeholders suitable to receive observations from the obs_space."""
    phs = restore_original_dimensions(keras.Input(shape=obs_space.shape), obs_space)
    if isinstance(phs, (tuple, list)):
        phs = [keras.Input(ph.shape[-1:]) for ph in phs]
    return phs


def action_input(action_space):
    """Create a keras.Input suitable to receive actions from the action_space."""
    return keras.Input(shape=action_space.shape)
