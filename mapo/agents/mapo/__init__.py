# pylint: disable=missing-docstring
from ray.rllib.utils import renamed_agent
from mapo.agents.mapo.mapo import MAPOTrainer, DEFAULT_CONFIG

MAPOAgent = renamed_agent(MAPOTrainer)

__all__ = ["MAPOTrainer", "MAPOAgent", "DEFAULT_CONFIG"]
