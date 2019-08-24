"""Tests regarding exploration features in TD3."""
# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import numpy as np
import scipy.stats as stats

from mapo.agents.td3.td3_policy import TD3TFPolicy


@pytest.fixture
def policy_config():
    return lambda env_name: {
        "env": env_name,
        # Expand bounds so that almost no distribution samples are clipped to range
        "env_config": {"action_dim": 1, "action_low": -100, "action_high": 100},
    }


def test_deterministic_evaluation(env_name, env_creator, policy_config):
    policy_config = policy_config(env_name)
    env = env_creator(env_name, config=policy_config["env_config"])
    ob_space, ac_space = env.observation_space, env.action_space
    policy = TD3TFPolicy(ob_space, ac_space, policy_config)

    obs = ob_space.sample()
    policy.evaluate(True)
    action1, _, _ = policy.compute_single_action(obs, [])
    action2, _, _ = policy.compute_single_action(obs, [])
    assert np.allclose(action1, action2)


@pytest.mark.skip
@pytest.mark.slow
def test_pure_exploration(env_name, env_creator, policy_config):
    policy_config = policy_config(env_name)
    env = env_creator(env_name, config=policy_config["env_config"])
    ob_space, ac_space = env.observation_space, env.action_space
    policy = TD3TFPolicy(ob_space, ac_space, policy_config)

    policy.set_pure_exploration_phase(True)
    obs = ob_space.sample()[None]

    def cdf(var):
        return stats.uniform.cdf(
            var, loc=ac_space.low, scale=ac_space.high - ac_space.low
        )

    def rvs(size=1):
        return np.squeeze(policy.compute_actions(obs.repeat(size, axis=0), [])[0])

    _, p_value = stats.kstest(rvs, cdf, N=10000)
    assert p_value >= 0.05


@pytest.mark.skip
@pytest.mark.slow
def test_iid_gaussian_exploration(env_name, env_creator, policy_config):
    policy_config = policy_config(env_name)
    env = env_creator(env_name, config=policy_config["env_config"])
    policy_config["exploration_noise_type"] = "gaussian"
    policy_config["exploration_gaussian_sigma"] = 0.5
    policy = TD3TFPolicy(env.observation_space, env.action_space, policy_config)

    obs = env.observation_space.sample()[None]
    policy.evaluate(True)
    loc = np.squeeze(policy.compute_single_action(obs[0], [])[0])
    policy.evaluate(False)
    policy.set_pure_exploration_phase(False)

    def cdf(var):
        return stats.norm.cdf(
            var, loc=loc, scale=policy_config["exploration_gaussian_sigma"]
        )

    def rvs(size=1):
        return np.squeeze(policy.compute_actions(obs.repeat(size, axis=0), [])[0])

    _, p_value = stats.kstest(rvs, cdf, N=10000)
    assert p_value >= 0.05
