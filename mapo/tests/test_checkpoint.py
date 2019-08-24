"""Tests for agent saving and loading."""
import pytest
import ray
from ray.tune.logger import NoopLogger
from ray.rllib.tests.test_checkpoint_restore import get_mean_action


@pytest.mark.slow
def test_checkpoint_restore(tmpdir, trainer_cls, env_name, env_creator):
    """
    Check if agents can be succesfully restored.

    Assumes the result of `compute_action` is deterministic for a given observation.
    """
    ray.init(ignore_reinit_error=True)

    def get_agent():
        config = {
            "train_batch_size": 100,
            "num_workers": 0,
            "env": env_name,
            "env_config": {"action_dim": 1},
        }

        def test_logger_creator(_):
            return NoopLogger(None, None)

        return trainer_cls(config=config, logger_creator=test_logger_creator)

    agent1 = get_agent()
    agent2 = get_agent()

    for _ in range(3):
        agent1.train()
    agent2.restore(agent1.save(tmpdir))

    obs = env_creator(env_name).observation_space.sample()
    mean_action1 = get_mean_action(agent1, obs)
    mean_action2 = get_mean_action(agent2, obs)
    assert abs(mean_action1 - mean_action2) <= 0.1
