"""Tests regarding exploration features in OffMAPO."""
import numpy as np
from ray.rllib.evaluation import RolloutWorker

from mapo.tests.mock_env import MockEnv
from mapo.agents.off_mapo.off_mapo_policy import OffMAPOTFPolicy


def test_evaluation():
    """Check if compute_actions is deterministic when evaluating."""
    worker = RolloutWorker(MockEnv, OffMAPOTFPolicy, policy_config={"evaluate": True})
    policy = worker.get_policy()

    for _ in range(3):
        policy.learn_on_batch(worker.sample())

    obs = MockEnv().observation_space.sample()
    action1, _, _ = policy.compute_single_action(obs, [])
    action2, _, _ = policy.compute_single_action(obs, [])
    assert np.allclose(action1, action2)
