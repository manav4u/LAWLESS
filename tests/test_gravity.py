import torch
import pytest

from lawless.physics.dynamics import rotate_gravity_90


def test_rotate_gravity_positive_90():
    g = torch.tensor([[0.0, 4.0], [4.0, 0.0]], dtype=torch.float64)
    out = rotate_gravity_90(g, 1)
    expected = torch.tensor([[-4.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    torch.testing.assert_close(out, expected)


def test_rotate_gravity_negative_90():
    g = torch.tensor([[0.0, 4.0]], dtype=torch.float64)
    out = rotate_gravity_90(g, -1)
    torch.testing.assert_close(out, torch.tensor([[4.0, 0.0]], dtype=torch.float64))


def test_rotate_gravity_rejects_bad_direction():
    with pytest.raises(ValueError):
        rotate_gravity_90(torch.tensor([[0.0, 1.0]]), 0)
