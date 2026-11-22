"""Actuator energy accounting."""
import torch

from .constants import PhysicsConfig


def actuator_energy_step(energy: torch.Tensor, throttle: torch.Tensor, torque_command: torch.Tensor, config: PhysicsConfig):
    """Return (energy_next, enabled_mask). Energy is measured in joules."""
    throttle = torch.as_tensor(throttle, device=energy.device, dtype=energy.dtype).clamp(-1.0, 1.0)
    torque_command = torch.as_tensor(torque_command, device=energy.device, dtype=energy.dtype).clamp(-1.0, 1.0)
    enabled = energy > 0
    power = config.thrust_power * throttle.abs() + config.torque_power * torque_command.abs()
    spent = power * config.dt * enabled.to(energy.dtype)
    energy_next = (energy - spent).clamp(min=0.0, max=config.max_energy)
    return energy_next, enabled
