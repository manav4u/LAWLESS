import torch

from lawless.physics.dynamics import linear_acceleration, angular_acceleration
from lawless.physics.constants import PhysicsConfig


def test_thrust_acceleration_exact_first_step():
    cfg = PhysicsConfig(linear_damping=0.0)
    velocity = torch.zeros((1, 2), dtype=torch.float64)
    angle = torch.zeros((1, 1), dtype=torch.float64)
    throttle = torch.ones((1, 1), dtype=torch.float64)
    gravity = torch.zeros((1, 2), dtype=torch.float64)
    acc = linear_acceleration(velocity, angle, throttle, gravity, cfg)
    torch.testing.assert_close(acc, torch.tensor([[12.0, 0.0]], dtype=torch.float64))


def test_reverse_thrust():
    cfg = PhysicsConfig(linear_damping=0.0)
    z2 = torch.zeros((1, 2), dtype=torch.float64)
    z1 = torch.zeros((1, 1), dtype=torch.float64)
    acc = linear_acceleration(z2, z1, -torch.ones_like(z1), z2, cfg)
    torch.testing.assert_close(acc, torch.tensor([[-12.0, 0.0]], dtype=torch.float64))


def test_gravity_and_damping():
    cfg = PhysicsConfig(max_thrust=0.001, linear_damping=0.15)
    velocity = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    angle = torch.zeros((1, 1), dtype=torch.float64)
    throttle = torch.zeros((1, 1), dtype=torch.float64)
    gravity = torch.tensor([[0.0, 4.0]], dtype=torch.float64)
    acc = linear_acceleration(velocity, angle, throttle, gravity, cfg)
    torch.testing.assert_close(acc, torch.tensor([[-0.3, 4.15]], dtype=torch.float64))


def test_angular_acceleration():
    cfg = PhysicsConfig(angular_damping=0.2)
    omega = torch.tensor([[1.0]], dtype=torch.float64)
    command = torch.tensor([[1.0]], dtype=torch.float64)
    alpha = angular_acceleration(omega, command, cfg)
    torch.testing.assert_close(alpha, torch.tensor([[3.8]], dtype=torch.float64))
