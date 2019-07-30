"""Tests regarding delayed policy updates in OffMAPO."""
# pylint: disable=missing-docstring
import numpy as np
from ray.rllib.evaluation import RolloutWorker

from mapo.tests.mock_env import MockEnv
from mapo.agents.off_mapo.off_mapo_policy import OffMAPOTFPolicy


def test_target_network_initialization():
    worker = RolloutWorker(MockEnv, OffMAPOTFPolicy, policy_config={"policy_delay": 2})
    policy = worker.get_policy()

    def get_main_target_vars():
        main_vars = policy.get_session().run(policy.main_variables)
        target_vars = policy.get_session().run(policy.target_variables)
        return zip(main_vars, target_vars)

    assert all([np.allclose(main, target) for main, target in get_main_target_vars()])
    policy.learn_on_batch(worker.sample())
    assert not all(
        [np.allclose(main, target) for main, target in get_main_target_vars()]
    )


def test_optimizer_global_step_update():
    # pylint: disable=protected-access
    worker = RolloutWorker(MockEnv, OffMAPOTFPolicy, policy_config={"policy_delay": 2})
    policy = worker.get_policy()

    for _ in range(10):
        policy.learn_on_batch(worker.sample())

    sess = policy.get_session()
    assert sess.run(policy.global_step) == 10
    assert sess.run(policy._actor_optimizer.iterations) == 5
    assert sess.run(policy._critic_optimizer.iterations) == 10


def test_actor_update_frequency():
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


def test_target_update_frequency():
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
