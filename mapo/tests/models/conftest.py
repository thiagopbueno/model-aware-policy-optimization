# pylint: disable=missing-docstring
import pytest
import numpy as np
from gym.spaces import Box


@pytest.fixture
def spaces():
    return (
        Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        # Action range must be limited so as to not break ActionSquashingLayer
        Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    )
