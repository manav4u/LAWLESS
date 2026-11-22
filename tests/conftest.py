import pytest
import torch

from lawless.physics.constants import PhysicsConfig


@pytest.fixture
def cfg():
    return PhysicsConfig()


@pytest.fixture
def dtype():
    return torch.float64
