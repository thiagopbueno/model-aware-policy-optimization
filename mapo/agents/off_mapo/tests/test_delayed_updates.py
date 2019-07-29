"""Tests regarding delayed policy updates in OffMAPO."""

import numpy as np
from ray.rllib.evaluation import RolloutWorker

from mapo.tests.mock_env import MockEnv
from mapo.agents.off_mapo.off_mapo_policy import OffMAPOTFPolicy


def test_update_optimizer_global_step():
    """Check that apply operations are run the correct number of times."""
    # pylint: disable=protected-access
    worker = RolloutWorker(MockEnv, OffMAPOTFPolicy, policy_config={"policy_delay": 2})
    policy = worker.get_policy()

    for _ in range(10):
        policy.learn_on_batch(worker.sample())

    sess = policy.get_session()
    assert sess.run(policy.global_step) == 10
    assert sess.run(policy._actor_optimizer.iterations) == 5
    assert sess.run(policy._critic_optimizer.iterations) == 10


def test_actor_updates():
    """Check that actor variables are only changed at the appropriate intervals."""
    worker = RolloutWorker(MockEnv, OffMAPOTFPolicy, policy_config={"policy_delay": 2})
    policy = worker.get_policy()

    def get_actor_vars():
        return policy.get_session().run(policy.policy_model.variables)

    for iteration in range(1, 7):
        before = get_actor_vars()
        policy.learn_on_batch(worker.sample())
        after = get_actor_vars()
        all_close = all([np.allclose(varb, vara) for varb, vara in zip(before, after)])
        if iteration % 2 == 0:
            assert not all_close
        else:
            assert all_close


def test_target_updates():
    """Check that target variables are only changed at the appropriate intervals."""
    worker = RolloutWorker(MockEnv, OffMAPOTFPolicy, policy_config={"policy_delay": 2})
    policy = worker.get_policy()

    def get_target_vars():
        return policy.get_session().run(policy.target_variables)

    for iteration in range(1, 7):
        before = get_target_vars()
        policy.learn_on_batch(worker.sample())
        after = get_target_vars()
        all_close = all([np.allclose(varb, vara) for varb, vara in zip(before, after)])
        if iteration % 2 == 0:
            assert not all_close
        else:
            assert all_close
