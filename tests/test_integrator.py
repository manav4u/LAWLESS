import math
import torch

from lawless.physics.integrator import integrate_linear, integrate_angular, wrap_angle


def test_semi_implicit_euler_reference_step():
    dt = 1.0 / 60.0
    pos = torch.zeros((1, 2), dtype=torch.float64)
    vel = torch.zeros((1, 2), dtype=torch.float64)
    acc = torch.tensor([[12.0, 0.0]], dtype=torch.float64)
    pos_next, vel_next = integrate_linear(pos, vel, acc, dt)
    torch.testing.assert_close(vel_next, torch.tensor([[0.2, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(pos_next, torch.tensor([[0.2 / 60.0, 0.0]], dtype=torch.float64))


def test_angle_wrap_interval():
    values = torch.tensor([[-4 * math.pi], [-math.pi], [math.pi], [7 * math.pi]], dtype=torch.float64)
    wrapped = wrap_angle(values)
    assert torch.all(wrapped >= -math.pi)
    assert torch.all(wrapped < math.pi)
    torch.testing.assert_close(wrapped[1], torch.tensor([-math.pi], dtype=torch.float64))
    torch.testing.assert_close(wrapped[2], torch.tensor([-math.pi], dtype=torch.float64))


def test_integrate_angular():
    angle = torch.zeros((1, 1), dtype=torch.float64)
    omega = torch.zeros_like(angle)
    alpha = torch.full_like(angle, 2.0)
    angle_next, omega_next = integrate_angular(angle, omega, alpha, 0.5)
    torch.testing.assert_close(omega_next, torch.tensor([[1.0]], dtype=torch.float64))
    torch.testing.assert_close(angle_next, torch.tensor([[0.5]], dtype=torch.float64))
