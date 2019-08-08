"""Exports MAPOTrainer."""

import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_CONFIG = with_common_config({})


MAPOTrainer = build_trainer(
    name="MAPO", default_policy=None, default_config=DEFAULT_CONFIG
)
