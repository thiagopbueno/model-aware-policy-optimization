# pylint: disable=missing-docstring,redefined-outer-name
import pytest
from ray.tune.registry import register_env, _global_registry, ENV_CREATOR

from mapo.agents.registry import ALGORITHMS
from mapo.tests.mock_env import MockEnv


@pytest.fixture(params=list(ALGORITHMS.values()))
def trainer_cls(request):
    return request.param()


@pytest.fixture(scope="session")
def env_name():
    register_env(MockEnv.__name__, lambda config=None: MockEnv(config))
    return MockEnv.__name__


@pytest.fixture(scope="session")
def env_creator(env_name):
    return _global_registry.get(ENV_CREATOR, env_name)


@pytest.fixture
def policy_cls(trainer_cls):
    return trainer_cls._policy  # pylint: disable=protected-access
