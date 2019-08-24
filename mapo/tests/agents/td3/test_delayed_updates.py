"""Tests regarding delayed policy updates in TD3."""
# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import numpy as np
from ray.rllib.evaluation import RolloutWorker

from mapo.agents.td3.td3_policy import TD3TFPolicy


@pytest.fixture
def worker(env_creator, env_name):
    return RolloutWorker(
        env_creator=env_creator,
        policy=TD3TFPolicy,
        policy_config={"policy_delay": 2, "env": env_name},
    )


def test_target_network_initialization(worker):
    policy = worker.get_policy()

    def get_main_target_vars():
        main_vars = policy.get_session().run(policy.model.variables())
        target_vars = policy.get_session().run(policy.target_model.variables())
        return zip(main_vars, target_vars)

    assert all([np.allclose(main, target) for main, target in get_main_target_vars()])
    policy.learn_on_batch(worker.sample())
    assert not all(
        [np.allclose(main, target) for main, target in get_main_target_vars()]
    )


def test_optimizer_global_step_update(worker):
    # pylint: disable=protected-access
    policy = worker.get_policy()

    for _ in range(10):
        policy.learn_on_batch(worker.sample())

    sess = policy.get_session()
    assert sess.run(policy.global_step) == 10
    assert sess.run(policy._actor_optimizer.iterations) == 5
    assert sess.run(policy._critic_optimizer.iterations) == 10


def test_actor_update_frequency(worker):
    policy = worker.get_policy()

    def get_actor_vars():
        return policy.get_session().run(policy.model.actor_variables)

    for iteration in range(1, 7):
        before = get_actor_vars()
        policy.learn_on_batch(worker.sample())
        after = get_actor_vars()
        all_close = all([np.allclose(varb, vara) for varb, vara in zip(before, after)])
        if iteration % 2 == 0:
            assert not all_close
        else:
            assert all_close


def test_target_update_frequency(worker):
    policy = worker.get_policy()

    def get_target_vars():
        return policy.get_session().run(policy.target_model.variables())

    for iteration in range(1, 7):
        before = get_target_vars()
        policy.learn_on_batch(worker.sample())
        after = get_target_vars()
        all_close = all([np.allclose(varb, vara) for varb, vara in zip(before, after)])
        if iteration % 2 == 0:
            assert not all_close
        else:
            assert all_close
