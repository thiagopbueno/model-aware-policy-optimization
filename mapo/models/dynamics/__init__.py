"""Collection of dynamics models."""

from mapo.models import obs_input, action_input
from mapo.models.dynamics.gaussian import GaussianDynamicsModel
from mapo.models.dynamics.gaussian_constant_stddev import (
    GaussianConstantStdDevDynamicsModel,
)

DYNAMICS_MODELS = {
    GaussianDynamicsModel.__name__: GaussianDynamicsModel,
    GaussianConstantStdDevDynamicsModel.__name__: GaussianConstantStdDevDynamicsModel,
}


def build_dynamics_model(obs_space, action_space, class_name=None, **config):
    """Returns the appropriate dynamics model based on config.

    Calls the dynamics model on input placeholders to initialize variables.
    """

    obs, action = obs_input(obs_space), action_input(action_space)

    if class_name in DYNAMICS_MODELS:
        model = DYNAMICS_MODELS[class_name](obs_space, action_space, **config)
        model([obs, action])
        return model

    raise ValueError(
        "Unknow dynamics model type '{}'. Try one of {}".format(
            class_name, list(DYNAMICS_MODELS.keys())
        )
    )
