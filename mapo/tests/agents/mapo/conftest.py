# pylint: disable=missing-docstring, redefined-outer-name
import pytest

from mapo.agents.mapo.mapo import MAPOTrainer
from mapo.agents.mapo.off_mapo import OffMAPOTrainer


@pytest.fixture(params=[MAPOTrainer, OffMAPOTrainer])
def trainer_cls(request):
    return request.param


@pytest.fixture
def policy_cls(trainer_cls):
    return trainer_cls._policy  # pylint: disable=protected-access


@pytest.fixture(scope="module")
def policy_cls_with_targets():
    return OffMAPOTrainer._policy  # pylint: disable=protected-access
