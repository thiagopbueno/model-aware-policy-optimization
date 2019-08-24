# pylint: disable=missing-docstring,redefined-outer-name
import pytest
from ray.tune.registry import register_env, _global_registry, ENV_CREATOR
from ray.rllib.models import ModelCatalog

from mapo.agents.registry import ALGORITHMS
from mapo.tests.mock_env import MockEnv, TimeAwareMockEnv


@pytest.fixture(params=list(ALGORITHMS.values()))
def trainer_cls(request):
    return request.param()


@pytest.fixture(scope="session", params=[MockEnv, TimeAwareMockEnv])
def env_name(request):
    env_cls = request.param
    register_env(env_cls.__name__, lambda config=None: env_cls(config))
    return env_cls.__name__


@pytest.fixture(scope="session")
def env_creator():
    return lambda name, config=None: _global_registry.get(ENV_CREATOR, name)(config)


@pytest.fixture
def spaces():
    return lambda env: (
        ModelCatalog.get_preprocessor_for_space(
            env.observation_space
        ).observation_space,
        env.action_space,
    )


@pytest.fixture
def policy_cls(trainer_cls):
    return trainer_cls._policy  # pylint: disable=protected-access
