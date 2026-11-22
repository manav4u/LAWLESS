"""Vectorized continuous-time acceleration calculations."""
import torch

from .constants import PhysicsConfig


def _as_like(value, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def linear_acceleration(
    velocity: torch.Tensor,
    angle: torch.Tensor,
    throttle: torch.Tensor,
    gravity: torch.Tensor,
    config: PhysicsConfig,
    *,
    enabled: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute acceleration in m/s^2. All leading batch dimensions broadcast."""
    throttle = _as_like(throttle, velocity).clamp(-1.0, 1.0)
    gravity = _as_like(gravity, velocity)
    if gravity.shape[-1] != 2:
        raise ValueError("gravity must have final dimension 2")
    direction = torch.cat((torch.cos(angle), torch.sin(angle)), dim=-1)
    if enabled is not None:
        throttle = throttle * enabled.to(velocity.dtype)
    thrust_acc = (config.max_thrust / config.mass) * throttle * direction
    return thrust_acc + gravity - config.linear_damping * velocity


def angular_acceleration(
    angular_velocity: torch.Tensor,
    torque_command: torch.Tensor,
    config: PhysicsConfig,
    *,
    enabled: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute angular acceleration in rad/s^2."""
    command = _as_like(torque_command, angular_velocity).clamp(-1.0, 1.0)
    if enabled is not None:
        command = command * enabled.to(angular_velocity.dtype)
    return (config.max_torque / config.inertia) * command - config.angular_damping * angular_velocity


def rotate_gravity_90(gravity: torch.Tensor, direction: torch.Tensor | int) -> torch.Tensor:
    """Rotate gravity by +90 or -90 degrees; direction is +1 or -1."""
    if gravity.shape[-1] != 2:
        raise ValueError("gravity must have final dimension 2")
    sign = torch.as_tensor(direction, device=gravity.device, dtype=gravity.dtype)
    if torch.any((sign != 1) & (sign != -1)):
        raise ValueError("direction must be +1 or -1")
    x, y = gravity[..., 0:1], gravity[..., 1:2]
    return torch.cat((-sign * y, sign * x), dim=-1)
