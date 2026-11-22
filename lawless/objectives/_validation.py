"""Shared validation for objective interactions."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from lawless.physics.constants import PhysicsConfig
from lawless.world.collision import CollisionResolution
from lawless.world.config import ObjectiveConfig
from lawless.world.episode_state import AgentEpisodeState
from lawless.physics.state import AgentState


def _validate_mask(
    name: str,
    value: Tensor,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must have torch.bool dtype")
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{name} shape {tuple(value.shape)} != expected {shape}"
        )
    if value.device != device:
        raise ValueError(f"{name} must be on device {device}")


def _validate_objective_config(
    config: ObjectiveConfig | None,
    *,
    core_count: int,
) -> None:
    if config is None:
        return
    if not isinstance(config, ObjectiveConfig):
        raise TypeError("objective_config must be an ObjectiveConfig")
    config.validate()
    if config.core_count != core_count:
        raise ValueError(
            "objective_config.core_count must match generated core count"
        )


def _validate_agent_state_tensors(
    name: str,
    physics: AgentState,
) -> None:
    """Validate tensor structure and finite floating fields before shape APIs."""

    if not isinstance(physics, AgentState):
        raise TypeError(f"{name} must be an AgentState")

    field_names = (
        "position",
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
        "alive",
    )

    # Guard every field before validate_shapes accesses .shape or .dtype.
    for field_name in field_names:
        value = getattr(physics, field_name)
        if not isinstance(value, Tensor):
            raise TypeError(f"{name}.{field_name} must be a torch.Tensor")

    position = physics.position
    if position.ndim < 1 or position.shape[-1] != 2:
        raise ValueError(f"{name}.position must have shape [..., 2]")
    if position.dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f"{name}.position must use float32 or float64"
        )

    expected_shapes = {
        "position": tuple(position.shape),
        "velocity": tuple(position.shape),
        "angle": tuple(position.shape[:-1]) + (1,),
        "angular_velocity": tuple(position.shape[:-1]) + (1,),
        "energy": tuple(position.shape[:-1]) + (1,),
        "alive": tuple(position.shape[:-1]) + (1,),
    }

    for field_name in field_names:
        value = getattr(physics, field_name)

        if tuple(value.shape) != expected_shapes[field_name]:
            raise ValueError(
                f"{name}.{field_name} shape {tuple(value.shape)} "
                f"!= expected {expected_shapes[field_name]}"
            )

        if value.device != position.device:
            raise ValueError(
                f"{name}.{field_name} must be on {position.device}"
            )

    for field_name in (
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
    ):
        value = getattr(physics, field_name)
        if value.dtype != position.dtype:
            raise ValueError(
                f"{name}.{field_name} must use {position.dtype}"
            )

    if physics.alive.dtype != torch.bool:
        raise TypeError(f"{name}.alive must have torch.bool dtype")

    for field_name in (
        "position",
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
    ):
        value = getattr(physics, field_name)
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(
                f"{name}.{field_name} must contain only finite values"
            )


def _validate_episode_and_resolution(
    episode_state: AgentEpisodeState,
    resolution: CollisionResolution,
    *,
    objective_config: ObjectiveConfig | None = None,
) -> None:
    """Validate state, post-resolution physics, reports, and layout alignment."""

    if not isinstance(episode_state, AgentEpisodeState):
        raise TypeError("episode_state must be an AgentEpisodeState")
    if not isinstance(resolution, CollisionResolution):
        raise TypeError("resolution must be a CollisionResolution")

    state_physics = episode_state.physics
    resolved_physics = resolution.physics

    _validate_agent_state_tensors(
        "episode_state.physics",
        state_physics,
    )
    _validate_agent_state_tensors(
        "resolution.physics",
        resolved_physics,
    )

    # Retain the accepted AgentState shape validation after the explicit
    # tensor-structure guards above.
    state_physics.validate_shapes()
    resolved_physics.validate_shapes()

    batch_shape = tuple(episode_state.batch_shape)
    mask_shape = batch_shape + (1,)
    device = state_physics.position.device
    dtype = state_physics.position.dtype

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("episode physics must use float32 or float64")

    if tuple(resolved_physics.batch_shape) != batch_shape:
        raise ValueError("resolution.physics batch shape does not match episode")

    for name in (
        "position",
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
    ):
        value = getattr(resolved_physics, name)
        reference = getattr(state_physics, name)
        if value.device != device:
            raise ValueError(f"resolution.physics.{name} is on the wrong device")
        if value.dtype != dtype:
            raise ValueError(f"resolution.physics.{name} has the wrong dtype")
        if value.shape != reference.shape:
            raise ValueError(f"resolution.physics.{name} has the wrong shape")

    _validate_mask(
        "resolution.physics.alive",
        resolved_physics.alive,
        shape=mask_shape,
        device=device,
    )
    _validate_mask(
        "episode_state.physics.alive",
        state_physics.alive,
        shape=mask_shape,
        device=device,
    )

    if not torch.equal(resolved_physics.alive, state_physics.alive):
        raise ValueError("resolution must preserve the episode alive mask")

    _validate_mask(
        "resolution.contact",
        resolution.contact,
        shape=mask_shape,
        device=device,
    )
    _validate_mask(
        "resolution.unresolved_contact",
        resolution.unresolved_contact,
        shape=mask_shape,
        device=device,
    )
    _validate_mask(
        "resolution.boundary_escape",
        resolution.boundary_escape,
        shape=mask_shape,
        device=device,
    )

    _validate_mask(
        "episode_state.extracted",
        episode_state.embodiment.extracted,
        shape=mask_shape,
        device=device,
    )
    _validate_mask(
        "episode_state.failed",
        episode_state.embodiment.failed,
        shape=mask_shape,
        device=device,
    )

    world = episode_state.generated_world
    if world.core_positions.device != device:
        raise ValueError("generated core positions are on the wrong device")
    if world.core_radii.device != device:
        raise ValueError("generated core radii are on the wrong device")
    if world.core_positions.dtype != dtype:
        raise ValueError("generated core positions have the wrong dtype")
    if world.core_radii.dtype != dtype:
        raise ValueError("generated core radii have the wrong dtype")

    if episode_state.objectives.device != device:
        raise ValueError("objective state is on the wrong device")
    if tuple(episode_state.objectives.batch_shape) != batch_shape:
        raise ValueError("objective state batch shape does not match episode")
    if episode_state.objectives.core_count != world.core_positions.shape[0]:
        raise ValueError("objective state core count does not match generated world")

    _validate_objective_config(
        objective_config,
        core_count=int(world.core_positions.shape[0]),
    )


def _positive_scalar_in_dtype(
    name: str,
    value: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Return a positive finite scalar represented safely in the target dtype."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")

    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc

    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")

    try:
        represented = torch.tensor(numeric, dtype=dtype, device=device)
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} cannot be represented in {dtype}"
        ) from exc

    if not bool(torch.isfinite(represented).item()):
        raise ValueError(f"{name} is not finite in {dtype}")
    if not bool((represented > 0).item()):
        raise ValueError(f"{name} is not positive in {dtype}")

    return represented


def _energy_parameters(
    objective_config: ObjectiveConfig,
    physics_config: PhysicsConfig,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    if not isinstance(objective_config, ObjectiveConfig):
        raise TypeError("objective_config must be an ObjectiveConfig")
    if not isinstance(physics_config, PhysicsConfig):
        raise TypeError("physics_config must be a PhysicsConfig")

    objective_config.validate()

    core_energy = _positive_scalar_in_dtype(
        "objective_config.core_energy_j",
        objective_config.core_energy_j,
        dtype=dtype,
        device=device,
    )
    max_energy = _positive_scalar_in_dtype(
        "physics_config.max_energy",
        physics_config.max_energy,
        dtype=dtype,
        device=device,
    )

    return core_energy, max_energy


def _validate_input_energy(
    energy: Tensor,
    *,
    max_energy: Tensor,
) -> None:
    """Require finite input energy in the inclusive interval [0, max_energy]."""

    if not bool(torch.isfinite(energy).all().item()):
        raise ValueError("input physics.energy must be finite")
    if bool((energy < 0).any().item()):
        raise ValueError("input physics.energy must be nonnegative")
    if bool((energy > max_energy).any().item()):
        raise ValueError("input physics.energy must not exceed max_energy")
