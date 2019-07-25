"""Tests for agent-environment interface."""
import pytest

from mapo.tests.mock_env import MockEnv
from mapo.agents.registry import ALGORITHMS


@pytest.mark.parametrize("get_trainer", list(ALGORITHMS.values()))
def test_compute_single_action(get_trainer):
    """Test if policy returns single action from the correct space."""
    env = MockEnv()
    trainer = get_trainer()
    # pylint: disable=protected-access
    policy = trainer._policy(env.observation_space, env.action_space, {})

    obs = env.reset()
    # the list is a placeholder for RNN state inputs
    action, _, _ = policy.compute_single_action(obs, [])

    assert action in env.action_space
