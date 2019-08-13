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
        # Learning rate for the dynamics optimizer.
        "dynamics_lr": 1e-3,
        # Learning rate for the critic (Q-function) optimizer.
        "critic_lr": 1e-3,
        # Learning rate for the actor (policy) optimizer.
        "actor_lr": 1e-3,
        # delayed policy update
        "policy_delay": 1,
        # Which model-learning optimization to use
        # Valid values: "mle" (Maximum Likelihood Estimation),
        # "pg-aware" (DPG-aware loss function),
        "model_loss": "mle",
        # === Resources ===
        # Number of actors used for parallelism
        "num_workers": 0,
    }
)


MAPOTrainer = build_trainer(
    name="MAPO", default_policy=MAPOTFPolicy, default_config=DEFAULT_CONFIG
)
