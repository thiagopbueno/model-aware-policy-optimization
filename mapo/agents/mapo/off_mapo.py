"""Driver for off-policy variant of MAPO."""

import logging

from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.optimizers import SyncReplayOptimizer
from ray.rllib.utils import merge_dicts
from mapo.agents.mapo.mapo import validate_config, DEFAULT_CONFIG as BASE_CONFIG
from mapo.agents.mapo.off_mapo_policy import OffMAPOTFPolicy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_CONFIG = merge_dicts(
    BASE_CONFIG,
    {
        # === Model ===
        # twin Q-net
        "twin_q": True,
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
        # === Optimization ===
        # Update the target by \tau * policy + (1-\tau) * target_policy
        "tau": 0.005,
        # target policy smoothing
        "smooth_target_policy": True,
        # gaussian stddev of target action noise for smoothing
        "target_noise": 0.2,
        # target noise limit (bound)
        "target_noise_clip": 0.5,
        # delayed policy update
        "policy_delay": 2,
        # How many environment steps to take before learning starts.
        "learning_starts": 0,
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
    },
)


def make_policy_optimizer(worker_set, config):
    """Create SyncReplayOptimizer."""
    return SyncReplayOptimizer(
        worker_set,
        learning_starts=config["learning_starts"],
        buffer_size=config["buffer_size"],
        prioritized_replay=False,
        train_batch_size=config["train_batch_size"],
        sample_batch_size=config["sample_batch_size"],
    )


def add_pure_exploration_phase(trainer):
    """
    Set all policies to pure exploration until `pure_exploration_steps` have elapsed.
    """
    global_timestep = trainer.optimizer.num_steps_sampled
    pure_expl_steps = trainer.config["pure_exploration_steps"]
    if pure_expl_steps:
        # tell workers whether they should do pure exploration
        only_explore = global_timestep < pure_expl_steps
        trainer.workers.local_worker().foreach_trainable_policy(
            lambda p, _: p.set_pure_exploration_phase(only_explore)
        )
        for worker in trainer.workers.remote_workers():
            worker.foreach_trainable_policy.remote(
                lambda p, _: p.set_pure_exploration_phase(only_explore)
            )


OffMAPOTrainer = build_trainer(
    name="OffMAPO",
    default_policy=OffMAPOTFPolicy,
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    make_policy_optimizer=make_policy_optimizer,
    before_train_step=add_pure_exploration_phase,
)
