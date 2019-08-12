# pylint: disable=missing-docstring
from ray.rllib.models.catalog import ModelCatalog

from mapo.agents.td3.td3 import TD3Trainer, DEFAULT_CONFIG
from mapo.agents.td3.td3_model import TD3Model

ModelCatalog.register_custom_model("td3_model", TD3Model)


__all__ = ["TD3Trainer", "DEFAULT_CONFIG"]
