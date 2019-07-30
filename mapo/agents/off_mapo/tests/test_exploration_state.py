"""Tests regarding exploration features in OffMAPO."""
import numpy as np
import scipy.stats as stats
from ray.rllib.evaluation import RolloutWorker

from mapo.tests.mock_env import MockEnv
from mapo.agents.off_mapo.off_mapo_policy import OffMAPOTFPolicy


def test_evaluation():
    """Check if compute_actions is deterministic when evaluating."""
    worker = RolloutWorker(MockEnv, OffMAPOTFPolicy)
    policy = worker.get_policy()
    policy.evaluate(True)

    for _ in range(3):
        policy.learn_on_batch(worker.sample())

    obs = MockEnv().observation_space.sample()
    action1, _, _ = policy.compute_single_action(obs, [])
    action2, _, _ = policy.compute_single_action(obs, [])
    assert np.allclose(action1, action2)


def test_pure_exploration():
    """Check if action distribution is uniform using the Kolmogorov-Smirnov test."""
    env = MockEnv({"action_dim": 1})
    ob_space, ac_space = env.observation_space, env.action_space
    policy = OffMAPOTFPolicy(ob_space, ac_space, {})
    policy.set_pure_exploration_phase(True)

    obs = ob_space.sample()[None]

    def cdf(var):
        return stats.uniform.cdf(
            var, loc=ac_space.low, scale=ac_space.high - ac_space.low
        )

    def rvs(size=1):
        return np.squeeze(policy.compute_actions(obs.repeat(size, axis=0), [])[0])

    _, p_value = stats.kstest(rvs, cdf, N=2000)
    assert p_value >= 0.1


def test_gaussian_exploration():
    """Check if action distribution is normal using the Kolmogorov-Smirnov test."""
    env = MockEnv({"action_dim": 1})
    policy = OffMAPOTFPolicy(
        env.observation_space, env.action_space, {"exploration_gaussian_sigma": 0.5}
    )

    obs = env.observation_space.sample()
    policy.evaluate(True)
    loc = np.squeeze(policy.compute_single_action(obs, [])[0])
    policy.evaluate(False)

    def cdf(var):
        return stats.norm.cdf(
            var, loc=loc, scale=policy.config["exploration_gaussian_sigma"]
        )

    def rvs(size=1):
        return np.squeeze(policy.compute_actions(obs[None].repeat(size, axis=0), [])[0])

    _, p_value = stats.kstest(rvs, cdf, N=2000)
    assert p_value >= 0.1
