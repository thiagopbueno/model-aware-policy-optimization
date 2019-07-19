"""Exports MAPOTrainer."""

import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy
from mapo.models.qnetwork import QNETWORK_DEFAULTS
from mapo.models.policy_network import POLICY_NETWORK_DEFAULTS

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_CONFIG = with_common_config(
    {
        # === Model ===
        # Model configuration for the actor network
        "actor_model": POLICY_NETWORK_DEFAULTS,
        # Model configuration for the critic network
        "critic_model": QNETWORK_DEFAULTS,
        # === Resources ===
        # Number of actors used for parallelism
        "num_workers": 0,
    }
)


MAPOTrainer = build_trainer(
    name="MAPO", default_policy=MAPOTFPolicy, default_config=DEFAULT_CONFIG
)
