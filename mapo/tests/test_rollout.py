"""Tests for agent-environment interface."""


def test_compute_single_action(policy_cls, env_name, env_creator, spaces):
    """Test if policy returns single action from the correct space."""
    # pylint: disable=protected-access
    env = env_creator(env_name)
    ob_space, ac_space = spaces(env)
    policy = policy_cls(ob_space, ac_space, {"env": env_name})

    obs = ob_space.sample()
    # the list is a placeholder for RNN state inputs
    action, _, _ = policy.compute_single_action(obs, [])

    assert action in env.action_space
