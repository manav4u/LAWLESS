import pytest
import torch

from lawless.physics.state import make_state


def test_make_state_shapes_and_dtypes():
    state = make_state((3, 5), dtype=torch.float64)
    assert state.position.shape == (3, 5, 2)
    assert state.velocity.shape == (3, 5, 2)
    assert state.angle.shape == (3, 5, 1)
    assert state.angular_velocity.shape == (3, 5, 1)
    assert state.energy.shape == (3, 5, 1)
    assert state.alive.shape == (3, 5, 1)
    assert state.position.dtype == torch.float64
    assert state.alive.dtype == torch.bool


def test_make_state_rejects_invalid_shape():
    with pytest.raises(ValueError):
        make_state((2, 0))
