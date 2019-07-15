"""Exports MAPOTrainer."""

import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_CONFIG = with_common_config(
    {
        "use_state_preprocessor": False,
        "actor_hiddens": [64, 64],
        "actor_hidden_activation": "relu",
        "critic_hiddens": [64, 64],
        "critic_hidden_activation": "relu",
    }
)


MAPOTrainer = build_trainer(
    name="MAPO", default_policy=MAPOTFPolicy, default_config=DEFAULT_CONFIG
)
