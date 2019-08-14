"""Tests regarding exploration features in TD3."""
# pylint: disable=missing-docstring
import numpy as np
import scipy.stats as stats
from ray.rllib.evaluation import RolloutWorker

from mapo.tests.mock_env import MockEnv
from mapo.agents.td3.td3_policy import TD3TFPolicy


def test_deterministic_evaluation():
    worker = RolloutWorker(MockEnv, TD3TFPolicy)
    policy = worker.get_policy()
    policy.evaluate(True)

    for _ in range(3):
        policy.learn_on_batch(worker.sample())

    obs = MockEnv().observation_space.sample()
    action1, _, _ = policy.compute_single_action(obs, [])
    action2, _, _ = policy.compute_single_action(obs, [])
    assert np.allclose(action1, action2)


def test_pure_exploration():
    env = MockEnv({"action_dim": 1})
    ob_space, ac_space = env.observation_space, env.action_space
    policy = TD3TFPolicy(ob_space, ac_space, {})
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


def test_iid_gaussian_exploration():
    # Expand bounds so that almost no distribution samples are clipped to range
    env = MockEnv({"action_dim": 1, "action_low": -100, "action_high": 100})
    policy = TD3TFPolicy(
        env.observation_space,
        env.action_space,
        {"exploration_noise_type": "gaussian", "exploration_gaussian_sigma": 0.5},
    )

    obs = env.observation_space.sample()[None]
    policy.evaluate(True)
    loc = np.squeeze(policy.compute_single_action(obs[0], [])[0])
    policy.evaluate(False)
    policy.set_pure_exploration_phase(False)

    def cdf(var):
        return stats.norm.cdf(
            var, loc=loc, scale=policy.config["exploration_gaussian_sigma"]
        )

    def rvs(size=1):
        return np.squeeze(policy.compute_actions(obs.repeat(size, axis=0), [])[0])

    _, p_value = stats.kstest(rvs, cdf, N=10000)
    assert p_value >= 0.05
