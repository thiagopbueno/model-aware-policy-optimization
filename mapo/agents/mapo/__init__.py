# pylint: disable=missing-docstring
from ray.rllib.models.catalog import ModelCatalog

from mapo.agents.mapo.mapo import MAPOTrainer, DEFAULT_CONFIG
from mapo.agents.mapo.off_mapo import OffMAPOTrainer
from mapo.agents.mapo.mapo_model import MAPOModel

ModelCatalog.register_custom_model("mapo_model", MAPOModel)


__all__ = ["MAPOTrainer", "OffMAPOTrainer", "DEFAULT_CONFIG"]
