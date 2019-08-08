"""Exports MAPOTrainer."""

import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy
from mapo.models.q_function import DEFAULT_CONFIG as ACTOR_MODEL_CONFIG
from mapo.models.policy import DEFAULT_CONFIG as CRITIC_MODEL_CONFIG

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_CONFIG = with_common_config(
    {
        # === Model ===
        "actor_model": ACTOR_MODEL_CONFIG,
        "critic_model": CRITIC_MODEL_CONFIG,
        # === Optimization ===
        # Learning rate for the critic (Q-function) optimizer.
        "critic_lr": 1e-3,
        # Learning rate for the actor (policy) optimizer.
        "actor_lr": 1e-3,
        # delayed policy update
        "policy_delay": 1,
        # === Resources ===
        # Number of actors used for parallelism
        "num_workers": 0,
    }
)


MAPOTrainer = build_trainer(
    name="MAPO", default_policy=MAPOTFPolicy, default_config=DEFAULT_CONFIG
)
