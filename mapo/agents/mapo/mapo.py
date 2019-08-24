"""Exports MAPOTrainer."""

import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils import merge_dicts
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_CONFIG = with_common_config(
    {
        # === MAPO ===
        # How many samples to draw from the dynamics model
        # If this is set to 0, use the next state observed from the environment
        "branching_factor": 1,
        # Whether to use the env's dynamics to calculate the actor loss
        "use_true_dynamics": False,
        # Which model-learning optimization to use
        # Valid values: "mle" (Maximum Likelihood Estimation),
        # "pga" (DPG-aware loss function),
        "model_loss": "mle",
        # Which kernel metric to compare gradients in dynamics loss
        "kernel": "l2",
        # === Model ===
        # actor and critic network configuration
        "model": merge_dicts(
            MODEL_DEFAULTS,
            {
                "custom_model": "mapo_model",
                "custom_options": {
                    "actor": {"activation": "relu", "layers": [400, 300]},
                    "critic": {"activation": "relu", "layers": [400, 300]},
                    "dynamics": {"activation": "relu", "layers": [400, 300]},
                },
            },
        ),
        # === Optimization ===
        # Optimizers for actor, critic, and dynamics networks.
        "actor_optimizer": "Adam",
        "critic_optimizer": "Adam",
        "dynamics_optimizer": "Adam",
        # Number of updates for critic and dynamics for each actor update.
        "critic_sgd_iter": 80,
        "dynamics_sgd_iter": 80,
        # Learning rate for the dynamics optimizer.
        "dynamics_lr": 1e-3,
        # Learning rate for the critic (Q-function) optimizer.
        "critic_lr": 1e-3,
        # Learning rate for the actor (policy) optimizer.
        "actor_lr": 1e-3,
        # delayed dynamics update
        "dynamics_delay": 1,
        # delayed critic update
        "critic_delay": 1,
        # delayed policy update
        "actor_delay": 1,
        # === Resources ===
        # Number of actors used for parallelism
        "num_workers": 0,
        # === Debugging ===
        # Specify where experiences should be saved:
        #  - None: don't save any experiences
        #  - "logdir" to save to the agent log dir
        #  - a path/URI to save to a custom output directory (e.g., "s3://bucket/")
        #  - a function that returns a rllib.offline.OutputWriter
        "output": None,
        # What sample batch columns to LZ4 compress in the output data.
        # RLlib's default is ["obs", "new_obs"], which saves space but makes the
        # output unreadable.
        "output_compress_columns": [],
    }
)


def validate_config(config):
    """Check for incorrect choices in config."""
    assert config["model"]["custom_model"] == "mapo_model", "MAPO depends on MAPOModel"
    assert config["model_loss"] in {
        "mle",
        "pga",
    }, "Unknown model_loss '{}' (try 'mle' or 'pga')".format(config["model_loss"])


MAPOTrainer = build_trainer(
    name="MAPO",
    default_policy=MAPOTFPolicy,
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
)
