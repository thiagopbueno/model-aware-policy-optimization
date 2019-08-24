# pylint: disable=missing-docstring, redefined-outer-name
import pytest
from ray.rllib.utils import merge_dicts
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.policy.policy import LEARNER_STATS_KEY


@pytest.fixture(params=[True, False])
def env_config(request):
    return {"fixed_state": request.param}


@pytest.fixture
def policy_cls_and_config(trainer_cls):
    # pylint: disable=protected-access
    return trainer_cls._policy, trainer_cls._default_config


@pytest.fixture
def model_config():
    return {
        "model": {"custom_options": {"dynamics": {"layers": [10]}}},
        "model_loss": "mle",
    }


@pytest.fixture
def worker(env_name, env_creator, policy_cls_and_config, model_config):
    policy_cls, policy_config = policy_cls_and_config
    return RolloutWorker(
        env_creator=lambda: env_creator(env_name),
        policy=policy_cls,
        policy_config=merge_dicts(policy_config, model_config),
    )


@pytest.mark.skip
def test_model_learns_deterministic_state(worker):
    policy = worker.get_policy()

    info = policy.learn_on_batch(worker.sample())
    initial_dynamics_loss = info[LEARNER_STATS_KEY]["dynamics_loss"]
    for _ in range(30):
        new_info = policy.learn_on_batch(worker.sample())
    assert new_info[LEARNER_STATS_KEY]["dynamics_loss"] <= initial_dynamics_loss
