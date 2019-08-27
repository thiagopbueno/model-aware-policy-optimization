"""Exports MAPOTrainer."""
import os.path as osp
import logging

import tensorflow as tf
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils import merge_dicts
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_CONFIG = with_common_config(
    {
        # === MAPO ===
        # How many samples to draw from the dynamics model
        # If this is set to 0, use the next state observed from the environment
        "branching_factor": 1,
        # Whether to use the env's dynamics to calculate the actor loss
        "use_true_dynamics": False,
        # Which model-learning optimization to use
        # Valid values: "mle" (Maximum Likelihood Estimation),
        # "pga" (DPG-aware loss function),
        "model_loss": "mle",
        # Which kernel metric to compare gradients in dynamics loss
        "kernel": "l2",
        # Which gradient estimator to use in model-aware dpg
        # Valid values: "sf" (score function), "pd" (pathwise derivative)
        # Warning: "pd" is incompatible with a branching factor of 0, for now
        "madpg_estimator": "sf",
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
        # Optimizers for actor, critic, and dynamics networks.
        "actor_optimizer": "Adam",
        "critic_optimizer": "Adam",
        "dynamics_optimizer": "Adam",
        # Clip gradients to be within this norm
        "max_grad_norm": float("inf"),
        # Whether to update components using separate "sgd_iter"s for each or
        # apply "delayed" updates to each.
        "apply_gradients": "delayed",
        # Number of updates for critic and dynamics for each actor update.
        "critic_sgd_iter": 80,
        "dynamics_sgd_iter": 80,
        # Learning rate for the dynamics optimizer.
        "dynamics_lr": 1e-3,
        # Learning rate for the critic (Q-function) optimizer.
        "critic_lr": 1e-3,
        # Learning rate for the actor (policy) optimizer.
        "actor_lr": 1e-3,
        # delayed dynamics update
        "dynamics_delay": 1,
        # delayed critic update
        "critic_delay": 1,
        # delayed policy update
        "actor_delay": 1,
        # Critic target update frequency
        "critic_target_update_freq": 1,
        # Actor target update frequency
        "actor_target_update_freq": 1,
        # Polyak averaging coefficient
        "tau": 1.0,
        # === Resources ===
        # Number of actors used for parallelism
        "num_workers": 0,
        # === Debugging ===
        "debug": False,
        # Specify where experiences should be saved:
        #  - None: don't save any experiences
        #  - "logdir" to save to the agent log dir
        #  - a path/URI to save to a custom output directory (e.g., "s3://bucket/")
        #  - a function that returns a rllib.offline.OutputWriter
        "output": None,
        # What sample batch columns to LZ4 compress in the output data.
        # RLlib's default is ["obs", "new_obs"], which saves space but makes the
        # output unreadable.
        "output_compress_columns": [],
    }
)


def validate_config(config):
    """Check for incorrect choices in config."""
    assert config["model"]["custom_model"] == "mapo_model", "MAPO depends on MAPOModel"
    assert config["model_loss"] in {
        "mle",
        "pga",
    }, "Unknown model_loss '{}' (try 'mle' or 'pga')".format(config["model_loss"])


def after_init(trainer):
    # pylint: disable=missing-docstring
    if trainer.config["debug"]:
        trainer.tf_writer = tf.compat.v1.summary.FileWriter(
            osp.join(trainer.logdir, "histograms")
        )


def after_optimizer_step(trainer, fetches):
    # pylint: disable=missing-docstring
    if trainer.config["debug"]:
        trainer.tf_writer.add_summary(
            fetches.pop("summaries"), trainer.optimizer.num_steps_sampled
        )
        trainer.tf_writer.flush()


MAPOTrainer = build_trainer(
    name="MAPO",
    default_policy=MAPOTFPolicy,
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    after_init=after_init,
    after_optimizer_step=after_optimizer_step,
)
