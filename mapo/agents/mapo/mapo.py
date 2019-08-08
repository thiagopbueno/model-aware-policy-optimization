"""Exports MAPOTrainer."""

import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_CONFIG = with_common_config(
    {
        # === Model ===
        # Apply a state preprocessor with spec given by the "model" config option
        # (like other RL algorithms). This is mostly useful if you have a weird
        # observation shape, like an image. Disabled by default.
        "use_state_preprocessor": False,
        # Postprocess the policy network model output with these hidden layers. If
        # use_state_preprocessor is False, then these will be the *only* hidden
        # layers in the network.
        "actor_hiddens": [64, 64],
        # Hidden layers activation of the postprocessing stage of the policy
        # network
        "actor_hidden_activation": "relu",
        # Postprocess the critic network model output with these hidden layers;
        # again, if use_state_preprocessor is True, then the state will be
        # preprocessed by the model specified with the "model" config option first.
        "critic_hiddens": [64, 64],
        # Hidden layers activation of the postprocessing state of the critic.
        "critic_hidden_activation": "relu",
        # === Resources ===
        # Number of actors used for parallelism
        "num_workers": 0,
        # === Optimization ===
        # Learning rate for the critic (Q-function) optimizer.
        "critic_lr": 1e-3,
        # Learning rate for the actor (policy) optimizer.
        "actor_lr": 1e-3,
    }
)


MAPOTrainer = build_trainer(
    name="MAPO", default_policy=MAPOTFPolicy, default_config=DEFAULT_CONFIG
)
