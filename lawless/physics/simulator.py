"""One deterministic, vectorized physics step."""
import torch

from .constants import PhysicsConfig
from .dynamics import linear_acceleration, angular_acceleration
from .energy import actuator_energy_step
from .integrator import integrate_linear, integrate_angular
from .state import AgentState


def step(
    state: AgentState,
    actions: torch.Tensor,
    gravity: torch.Tensor,
    config: PhysicsConfig,
) -> tuple[AgentState, dict[str, torch.Tensor]]:
    """Advance all agents by one fixed timestep.

    Args:
        state: leading dimensions [...], state fields documented in AgentState.
        actions: [..., 2], throttle and torque command in [-1, 1].
        gravity: [..., 2] or [2], m/s^2.
        config: immutable physics settings.

    Returns:
        A new AgentState and diagnostics (acceleration, angular_acceleration,
        energy_spent, actuators_enabled). Inputs are not mutated.
    """
    state.validate_shapes()
    if actions.shape != state.batch_shape + (2,):
        raise ValueError(f"actions shape must be {state.batch_shape + (2,)}, got {tuple(actions.shape)}")
    actions = actions.to(device=state.position.device, dtype=state.position.dtype).clamp(-1.0, 1.0)
    throttle = actions[..., 0:1]
    torque_command = actions[..., 1:2]

    energy_next, has_energy = actuator_energy_step(state.energy, throttle, torque_command, config)
    enabled = state.alive & has_energy
    accel = linear_acceleration(state.velocity, state.angle, throttle, gravity, config, enabled=enabled)
    ang_accel = angular_acceleration(state.angular_velocity, torque_command, config, enabled=enabled)

    pos_next, vel_next = integrate_linear(state.position, state.velocity, accel, config.dt)
    angle_next, omega_next = integrate_angular(state.angle, state.angular_velocity, ang_accel, config.dt)

    # Already-dead agents are frozen; newly exhausted agents complete this step,
    # then are marked inactive on subsequent steps.
    alive_next = state.alive & (energy_next > 0)
    pos_next = torch.where(state.alive, pos_next, state.position)
    vel_next = torch.where(state.alive, vel_next, state.velocity)
    angle_next = torch.where(state.alive, angle_next, state.angle)
    omega_next = torch.where(state.alive, omega_next, state.angular_velocity)

    next_state = AgentState(pos_next, vel_next, angle_next, omega_next, energy_next, alive_next)
    diagnostics = {
        "acceleration": accel,
        "angular_acceleration": ang_accel,
        "energy_spent": (state.energy - energy_next).clamp_min(0.0),
        "actuators_enabled": enabled,
    }
    return next_state, diagnostics
