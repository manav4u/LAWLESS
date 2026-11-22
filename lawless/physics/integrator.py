"""Semi-implicit Euler integration and angle utilities."""
import math
import torch


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap radians into [-pi, pi)."""
    two_pi = 2.0 * math.pi
    return torch.remainder(angle + math.pi, two_pi) - math.pi


def integrate_linear(position: torch.Tensor, velocity: torch.Tensor, acceleration: torch.Tensor, dt: float):
    velocity_next = velocity + acceleration * dt
    position_next = position + velocity_next * dt
    return position_next, velocity_next


def integrate_angular(angle: torch.Tensor, angular_velocity: torch.Tensor, angular_acceleration: torch.Tensor, dt: float):
    omega_next = angular_velocity + angular_acceleration * dt
    angle_next = wrap_angle(angle + omega_next * dt)
    return angle_next, omega_next
