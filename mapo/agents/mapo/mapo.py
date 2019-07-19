"""Exports MAPOTrainer."""

import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_CONFIG = with_common_config(
    {
        # === Model ===
        # Process the observation input tensors with these hidden layers.
        "actor_hiddens": [64, 64],
        # Hidden layers activation of the policy network
        "actor_hidden_activation": "relu",
        # Process the observation and action input tensors with these hidden layers.
        "critic_hiddens": [64, 64],
        # Hidden layers activation of the critic.
        "critic_hidden_activation": "relu",
        # === Resources ===
        # Number of actors used for parallelism
        "num_workers": 0,
    }
)


MAPOTrainer = build_trainer(
    name="MAPO", default_policy=MAPOTFPolicy, default_config=DEFAULT_CONFIG
)
