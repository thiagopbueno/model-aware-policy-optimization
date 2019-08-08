"""Tests for agent-environment interface."""
from mapo.tests.mock_env import MockEnv
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy


def test_compute_single_action():
    """Test if policy returns single action from the correct space."""
    env = MockEnv()
    policy = MAPOTFPolicy(env.observation_space, env.action_space, {})

    obs = env.reset()
    # the list is a placeholder for RNN state inputs
    action, _, _ = policy.compute_single_action(obs, [])

    assert action in env.action_space
