# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import numpy as np
from gym.spaces import Box

from mapo.tests.conftest import env_name, env_creator  # pylint: disable=unused-import
from mapo.agents.mapo.mapo import MAPOTrainer
from mapo.agents.mapo.off_mapo import OffMAPOTrainer


@pytest.fixture
def spaces():
    return (
        Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
        # Action range must be limited so as to not break ActionSquashingLayer
        Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
    )


@pytest.fixture(params=[MAPOTrainer, OffMAPOTrainer])
def trainer_cls(request):
    return request.param


@pytest.fixture
def policy_cls(trainer_cls):
    return trainer_cls._policy  # pylint: disable=protected-access


@pytest.fixture(scope="module")
def policy_cls_with_targets():
    return OffMAPOTrainer._policy  # pylint: disable=protected-access
