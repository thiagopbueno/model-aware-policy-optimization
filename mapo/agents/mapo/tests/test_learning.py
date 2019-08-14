# pylint: disable=missing-docstring, redefined-outer-name
from itertools import product

import pytest
from ray.rllib.utils import merge_dicts
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy.policy import LEARNER_STATS_KEY

from mapo.tests.mock_env import MockEnv
from mapo.agents.mapo.mapo_policy import (
    MAPOTFPolicy,
    get_default_config as onpolicy_config,
)
from mapo.agents.mapo.off_mapo_policy import (
    OffMAPOTFPolicy,
    get_default_config as offpolicy_config,
)


def get_env_makers():
    return [lambda _: MockEnv({"fixed_state": True})]


def get_policies_and_configs():
    return [(MAPOTFPolicy, onpolicy_config()), (OffMAPOTFPolicy, offpolicy_config())]


def get_config_mods():
    return [
        lambda config: merge_dicts(
            config,
            {
                "model": {"custom_options": {"dynamics": {"layers": []}}},
                "model_loss": "mle",
            },
        )
    ]


def get_worker_arguments():
    all_combinations = product(
        get_env_makers(), get_policies_and_configs(), get_config_mods()
    )
    return [
        (env, policy, config_fn(conf))
        for env, (policy, conf), config_fn in all_combinations
    ]


@pytest.mark.parametrize("worker_args", get_worker_arguments())
def test_model_learns_deterministic_state(worker_args):
    env_creator, policy_cls, policy_config = worker_args
    worker = RolloutWorker(env_creator, policy_cls, policy_config=policy_config)
    policy = worker.get_policy()

    info = policy.learn_on_batch(worker.sample())
    initial_dynamics_loss = info[LEARNER_STATS_KEY]["dynamics_loss"]
    for _ in range(10):
        new_info = policy.learn_on_batch(worker.sample())
    assert new_info[LEARNER_STATS_KEY]["dynamics_loss"] <= initial_dynamics_loss
