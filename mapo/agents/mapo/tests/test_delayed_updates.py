# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import numpy as np
from ray.rllib.evaluation import RolloutWorker

from mapo.tests.mock_env import MockEnv
from mapo.agents.mapo.mapo_policy import MAPOTFPolicy
from mapo.agents.mapo.off_mapo_policy import OffMAPOTFPolicy


def get_worker_kwargs():
    return [
        dict(
            env_creator=MockEnv,
            policy=policy_cls,
            policy_config={"actor_delay": 3, "critic_delay": 2},
        )
        for policy_cls in (MAPOTFPolicy, OffMAPOTFPolicy)
    ]


@pytest.fixture
def worker_with_targets():
    return RolloutWorker(**get_worker_kwargs()[1])


@pytest.mark.parametrize("worker_kwargs", get_worker_kwargs())
def test_optimizer_global_step_update(worker_kwargs):
    # pylint: disable=protected-access
    worker = RolloutWorker(**worker_kwargs)
    policy = worker.get_policy()

    for _ in range(10):
        policy.learn_on_batch(worker.sample())

    sess = policy.get_session()
    actor_delay, critic_delay = (
        policy.config["actor_delay"],
        policy.config["critic_delay"],
    )
    assert sess.run(policy.global_step) == 10
    assert sess.run(policy._optimizer.actor.iterations) == 10 // actor_delay
    assert sess.run(policy._optimizer.critic.iterations) == 10 // critic_delay


@pytest.mark.parametrize("worker_kwargs", get_worker_kwargs())
def test_actor_update_frequency(worker_kwargs):
    worker = RolloutWorker(**worker_kwargs)
    policy = worker.get_policy()

    def get_actor_vars():
        return policy.get_session().run(policy.model.actor_variables)

    for iteration in range(1, 7):
        before = get_actor_vars()
        policy.learn_on_batch(worker.sample())
        after = get_actor_vars()
        all_close = all([np.allclose(varb, vara) for varb, vara in zip(before, after)])
        if iteration % policy.config["actor_delay"] == 0:
            assert not all_close
        else:
            assert all_close


@pytest.mark.parametrize("worker_kwargs", get_worker_kwargs())
def test_critic_update_frequency(worker_kwargs):
    worker = RolloutWorker(**worker_kwargs)
    policy = worker.get_policy()

    def get_critic_vars():
        return policy.get_session().run(policy.model.critic_variables)

    for iteration in range(1, 7):
        before = get_critic_vars()
        policy.learn_on_batch(worker.sample())
        after = get_critic_vars()
        all_close = all([np.allclose(varb, vara) for varb, vara in zip(before, after)])
        if iteration % policy.config["critic_delay"] == 0:
            assert not all_close
        else:
            assert all_close


def test_target_network_initialization(worker_with_targets):
    worker = worker_with_targets
    policy = worker.get_policy()

    main_target_vars = policy.get_session().run(policy.model.main_and_target_variables)

    assert all([np.allclose(main, target) for main, target in main_target_vars])


def test_target_update_frequency(worker_with_targets):
    worker = worker_with_targets
    policy = worker.get_policy()

    def get_target_vars():
        sess = policy.get_session()
        return [
            target for main, target in sess.run(policy.model.main_and_target_variables)
        ]

    for iteration in range(1, 7):
        before = get_target_vars()
        policy.learn_on_batch(worker.sample())
        after = get_target_vars()
        all_close = all([np.allclose(varb, vara) for varb, vara in zip(before, after)])
        if iteration % policy.config["actor_delay"] == 0:
            assert not all_close
        else:
            assert all_close
