"""Exports TD3Trainer."""

import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.utils import merge_dicts
from mapo.agents.td3.td3_policy import TD3TFPolicy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

DEFAULT_CONFIG = with_common_config(
    {
        # === Model ===
        # actor and critic network configuration
        "model": merge_dicts(
            MODEL_DEFAULTS,
            {
                "custom_model": "td3_model",
                "custom_options": {
                    "actor": {"activation": "relu", "layers": [400, 300]},
                    "critic": {"activation": "relu", "layers": [400, 300]},
                },
            },
        ),
        # twin Q-net
        "twin_q": True,
        # === Optimization ===
        # Learning rate for the critic (Q-function) optimizer.
        "critic_lr": 1e-3,
        # Learning rate for the actor (policy) optimizer.
        "actor_lr": 1e-3,
        # Update the target by \tau * policy + (1-\tau) * target_policy
        "tau": 0.005,
        # target policy smoothing
        "smooth_target_policy": True,
        # gaussian stddev of target action noise for smoothing
        "target_noise": 0.2,
        # target noise limit (bound)
        "target_noise_clip": 0.5,
        # delayed policy update
        "actor_delay": 2,
        # How many environment steps to take before learning starts.
        "learning_starts": 0,
        # === Resources ===
        # Number of actors used for parallelism
        "num_workers": 0,
        # === Exploration ===
        # valid values: "ou" (time-correlated, like original DDPG paper),
        # "gaussian" (IID, like TD3 paper)
        "exploration_noise_type": "gaussian",
        # OU-noise scale; this can be used to scale down magnitude of OU noise
        # before adding to actions (requires "exploration_noise_type" to be "ou")
        "exploration_ou_noise_scale": 0.1,
        # theta for OU
        "exploration_ou_theta": 0.15,
        # sigma for OU
        "exploration_ou_sigma": 0.2,
        # gaussian stddev of act noise for exploration (requires
        # "exploration_noise_type" to be "gaussian")
        "exploration_gaussian_sigma": 0.1,
        # Until this many timesteps have elapsed, the agent's policy will be
        # ignored & it will instead take uniform random actions. Can be used in
        # conjunction with learning_starts (which controls when the first
        # optimization step happens) to decrease dependence of exploration &
        # optimization on initial policy parameters.
        "pure_exploration_steps": 1000,
        # === Replay buffer ===
        # Size of the replay buffer. Note that if async_updates is set, then
        # each worker will have a replay buffer of this size.
        "buffer_size": int(1e6),
        # === Execution ===
        # Update the replay buffer with this many samples at once. Note that this
        # setting applies per-worker if num_workers > 1.
        "sample_batch_size": 1,
        # Size of a batched sampled from replay buffer for training. Note that
        # if async_updates is set, then each worker returns gradients for a
        # batch of this size.
        "train_batch_size": 100,
        # Number of env steps to optimize for before returning
        "timesteps_per_iteration": 1000,
        # === Evaluation ===
        # Evaluate with every `evaluation_interval` training iterations.
        # The evaluation stats will be reported under the "evaluation" metric key.
        # Note that evaluation is currently not parallelized, and that for Ape-X
        # metrics are already only reported for the lowest epsilon workers.
        "evaluation_interval": None,
        # Number of episodes to run per evaluation period.
        "evaluation_num_episodes": 10,
        # Extra arguments to pass to evaluation workers.
        # Typical usage is to pass extra args to evaluation env creator
        # and to disable exploration by computing deterministic actions
        "evaluation_config": {"evaluate": True},
        # Turn of exploration noise when evaluating the policy
        "evaluate": False,
    }
)


TD3Trainer = build_trainer(
    name="OurTD3", default_policy=TD3TFPolicy, default_config=DEFAULT_CONFIG
)
