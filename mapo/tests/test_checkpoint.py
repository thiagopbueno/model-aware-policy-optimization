"""Tests for agent saving and loading."""
import pytest
import ray
from ray.tune import register_env
from ray.rllib.tests.test_checkpoint_restore import get_mean_action
from mapo.tests.mock_env import MockEnv
from mapo.agents.registry import ALGORITHMS


@pytest.mark.parametrize("get_trainer", list(ALGORITHMS.values()))
def test_checkpoint_restore(tmpdir, get_trainer):
    """
    Check if agents can be succesfully restored.

    Assumes the result of `compute_action` is deterministic for a given observation.
    """
    ray.init(ignore_reinit_error=True)
    register_env("test", lambda _: MockEnv({"action_dim": 1}))
    trainer = get_trainer()
    agent1 = trainer(env="test")
    agent2 = trainer(env="test")

    for _ in range(3):
        agent1.train()
    agent2.restore(agent1.save(tmpdir))

    env = MockEnv()
    obs = env.observation_space.sample()
    mean_action1 = get_mean_action(agent1, obs)
    mean_action2 = get_mean_action(agent2, obs)
    assert abs(mean_action1 - mean_action2) <= 0.1
