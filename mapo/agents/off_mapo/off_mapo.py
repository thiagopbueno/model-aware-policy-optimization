"""Driver for off-policy variant of MAPO."""

import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.optimizers import SyncReplayOptimizer
from mapo.agents.off_mapo.off_mapo_policy import OffMAPOTFPolicy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


DEFAULT_CONFIG = with_common_config(
    {
        # === Model ===
        # twin Q-net
        "twin_q": True,
        # Process the observation input tensors with these hidden layers.
        "actor_hiddens": [400, 300],
        # Hidden layers activation of the policy network
        "actor_hidden_activation": "relu",
        # Process the observation and action input tensors with these hidden layers.
        "critic_hiddens": [400, 300],
        # Hidden layers activation of the critic.
        "critic_hidden_activation": "relu",
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
        "policy_delay": 2,
        # How many environment steps to take before learning starts.
        "learning_starts": 1500,
        # === Resources ===
        # Number of actors used for parallelism
        "num_workers": 0,
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
    }
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


OffMAPOTrainer = build_trainer(
    name="OffMAPO",
    default_policy=OffMAPOTFPolicy,
    default_config=DEFAULT_CONFIG,
    make_policy_optimizer=make_policy_optimizer,
)
