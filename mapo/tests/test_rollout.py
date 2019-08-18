"""Tests for agent-environment interface."""


def test_compute_single_action(policy_cls, env_name, env_creator):
    """Test if policy returns single action from the correct space."""
    # pylint: disable=protected-access
    env = env_creator()
    policy = policy_cls(env.observation_space, env.action_space, {"env": env_name})

    obs = env.reset()
    # the list is a placeholder for RNN state inputs
    action, _, _ = policy.compute_single_action(obs, [])

    assert action in env.action_space
