"""Endpoint static-circle collision resolution for LAWLESS P1.3.

This module performs bounded, deterministic positional depenetration and
zero-restitution velocity response against shared static circles.

It is not a swept collision solver. It does not step an episode, consume
boundary failures, alter energy/history/objectives, or perform agent-agent
collision.

Reports (all shaped batch_shape + (1,)):
    contact: active agent encountered inclusive contact during resolution.
    unresolved_contact: residual penetration beyond tolerance remains.
    boundary_escape: input or resolved position is strictly outside bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from lawless.physics.state import AgentState
from lawless.world.world_state import WorldState


@dataclass(frozen=True)
class CollisionResolution:
    physics: AgentState
    contact: torch.Tensor
    unresolved_contact: torch.Tensor
    boundary_escape: torch.Tensor


def _tolerance(
    positions: torch.Tensor,
    agent_radius_m: float,
    layout_scale: torch.Tensor,
) -> torch.Tensor:
    """Scale-aware tolerance, independently computed for each agent."""
    local_scale = torch.maximum(
        positions.abs().amax(dim=-1, keepdim=True),
        torch.full_like(
            positions[..., :1],
            float(agent_radius_m),
        ),
    )
    local_scale = torch.maximum(local_scale, layout_scale)
    local_scale = torch.maximum(local_scale, torch.ones_like(local_scale))
    return 8.0 * torch.finfo(positions.dtype).eps * local_scale


def _layout_scale(world: WorldState, dtype: torch.dtype) -> torch.Tensor:
    """Shared layout/world scale; independent of the agent batch."""
    centers = world.obstacles.centers
    radii = world.obstacles.radii

    scale = torch.tensor(
        max(float(world.width_m), float(world.height_m), 1.0),
        dtype=dtype,
        device=centers.device,
    )

    if centers.numel():
        scale = torch.maximum(scale, centers.abs().amax())
    if radii.numel():
        scale = torch.maximum(scale, radii.abs().amax())

    return scale


def _strict_boundary_escape(
    positions: torch.Tensor,
    world: WorldState,
) -> torch.Tensor:
    """Strict outside-rectangle mask, with trailing scalar dimension."""
    x = positions[..., 0:1]
    y = positions[..., 1:2]

    return (
        (x < 0.0)
        | (x > world.width_m)
        | (y < 0.0)
        | (y > world.height_m)
    )


def _fallback_normal(
    obstacle_index: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Stable fallback direction for coincident centers.

    Obstacle index zero uses +x. Subsequent indices cycle through
    cardinal directions. This avoids random or device-dependent normals.
    """
    directions = (
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
    )
    direction = directions[obstacle_index % len(directions)]
    return torch.tensor(direction, dtype=dtype, device=device)


def resolve_static_obstacles(
    physics: AgentState,
    world: WorldState,
    *,
    agent_radius_m: float,
    max_passes: int = 8,
) -> CollisionResolution:
    """Resolve endpoint contact against ordered static circular obstacles.

    Position correction occurs only for penetration beyond tolerance.
    Inclusive contact classification also covers tangency and penetration
    within tolerance. Contact removes only inward normal velocity.

    ``max_passes`` is the maximum number of ordered obstacle passes.
    Residual penetration beyond tolerance is reported, not hidden.

    Inputs are not mutated. Dead agents are excluded and remain frozen.
    """
    if not isinstance(physics, AgentState):
        raise TypeError("physics must be an AgentState")

    # Structural guard before validate_shapes(), so malformed field types
    # produce a deliberate TypeError rather than AttributeError.
    field_names = (
        "position",
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
        "alive",
    )
    for name in field_names:
        if not isinstance(getattr(physics, name), torch.Tensor):
            raise TypeError(f"physics.{name} must be a torch.Tensor")

    if not isinstance(world, WorldState):
        raise TypeError("world must be a WorldState")

    physics.validate_shapes()

    position = physics.position
    velocity = physics.velocity
    dtype = position.dtype
    device = position.device

    if dtype not in (torch.float32, torch.float64):
        raise TypeError("position must use float32 or float64")

    for name in ("velocity", "angle", "angular_velocity", "energy"):
        value = getattr(physics, name)
        if value.dtype != dtype:
            raise TypeError(f"physics.{name} must use the position dtype")
        if value.device != device:
            raise ValueError(f"physics.{name} must use the position device")

    if physics.alive.device != device:
        raise ValueError("physics.alive must use the position device")

    obstacles = world.obstacles
    centers = obstacles.centers
    radii = obstacles.radii

    if centers.dtype != dtype or radii.dtype != dtype:
        raise TypeError("obstacle tensors must use the physics dtype")
    if centers.device != device or radii.device != device:
        raise ValueError("obstacle tensors must use the physics device")

    if (
        isinstance(agent_radius_m, bool)
        or not isinstance(agent_radius_m, (int, float))
    ):
        raise TypeError("agent_radius_m must be a real number")

    radius = float(agent_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("agent_radius_m must be finite and positive")

    represented_radius = torch.tensor(radius, dtype=dtype, device=device)
    if not bool(torch.isfinite(represented_radius)) or not bool(
        represented_radius > 0
    ):
        raise ValueError("agent_radius_m underflows or is non-finite in physics dtype")

    if isinstance(max_passes, bool) or not isinstance(max_passes, int):
        raise TypeError("max_passes must be an integer in [1, 8]")
    if not 1 <= max_passes <= 8:
        raise ValueError("max_passes must be in [1, 8]")

    radius = float(represented_radius.item())

    # Validate the input state without modifying it.
    for name in (
        "position",
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
    ):
        value = getattr(physics, name)
        if not torch.isfinite(value).all():
            raise ValueError(f"physics.{name} must contain only finite values")

    batch_shape = position.shape[:-1]
    flat_position = position.reshape(-1, 2).clone()
    flat_velocity = velocity.reshape(-1, 2).clone()
    flat_alive = physics.alive.reshape(-1, 1)

    layout_scale = _layout_scale(world, dtype)
    tolerance = _tolerance(flat_position, radius, layout_scale)

    contact = torch.zeros(
        (flat_position.shape[0], 1),
        dtype=torch.bool,
        device=device,
    )
    unresolved = torch.zeros_like(contact)

    # Report strict escape from the original state as well as any escape
    # introduced by correction. Collision resolution never clears it.
    boundary_escape = _strict_boundary_escape(flat_position, world).reshape(
        -1, 1
    ).clone()

    if centers.shape[0] == 0 or flat_position.shape[0] == 0:
        result_physics = physics.clone()
        return CollisionResolution(
            physics=result_physics,
            contact=contact.reshape(batch_shape + (1,)),
            unresolved_contact=unresolved.reshape(batch_shape + (1,)),
            boundary_escape=boundary_escape.reshape(batch_shape + (1,)),
        )

    combined_radii = radii + represented_radius

    # Ordered passes, preserving stable obstacle-index ordering.
    for _pass in range(max_passes):
        for obstacle_index in range(centers.shape[0]):
            obstacle_center = centers[obstacle_index]
            combined_radius = combined_radii[obstacle_index]

            delta = flat_position - obstacle_center
            distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)

            tau = _tolerance(flat_position, radius, layout_scale)

            # Inclusive contact. Dead agents do not participate.
            active = flat_alive
            touching = (distance <= combined_radius + tau) & active
            contact |= touching

            penetration = combined_radius - distance
            needs_correction = (penetration > tau) & active

            # Stable fallback for coincident center; otherwise use the
            # geometric radial normal.
            nonzero = distance > 0.0
            safe_distance = torch.where(
                nonzero,
                distance,
                torch.ones_like(distance),
            )
            normal = delta / safe_distance

            fallback = _fallback_normal(
                obstacle_index,
                dtype=dtype,
                device=device,
            ).expand_as(normal)
            normal = torch.where(nonzero.expand_as(normal), normal, fallback)

            correction_distance = torch.where(
                needs_correction,
                penetration,
                torch.zeros_like(penetration),
            )
            flat_position = flat_position + correction_distance * normal

            # Zero restitution: remove only inward normal velocity.
            vn = (flat_velocity * normal).sum(dim=-1, keepdim=True)
            remove_inward = (vn < 0.0) & touching
            flat_velocity = flat_velocity - torch.where(
                remove_inward,
                vn,
                torch.zeros_like(vn),
            ) * normal

        # Keep strict boundary escape persistent during all passes.
        boundary_escape |= _strict_boundary_escape(
            flat_position, world
        ).reshape(-1, 1)

    # Final residual check. Contact and unresolved are intentionally distinct.
    for obstacle_index in range(centers.shape[0]):
        delta = flat_position - centers[obstacle_index]
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        tau = _tolerance(flat_position, radius, layout_scale)

        residual_penetration = (
            distance < combined_radii[obstacle_index] - tau
        ) & flat_alive
        unresolved |= residual_penetration

        # Include any final contact not already observed during a pass.
        contact |= (
            distance <= combined_radii[obstacle_index] + tau
        ) & flat_alive

    # Construct fresh physics storage. Non-resolved physical fields remain
    # unchanged. Dead entries are restored exactly.
    final_position = flat_position.reshape(batch_shape + (2,))
    final_velocity = flat_velocity.reshape(batch_shape + (2,))

    final_position = torch.where(
        physics.alive.expand_as(final_position),
        final_position,
        physics.position,
    )
    final_velocity = torch.where(
        physics.alive.expand_as(final_velocity),
        final_velocity,
        physics.velocity,
    )

    result_physics = AgentState(
        position=final_position,
        velocity=final_velocity,
        angle=physics.angle.clone(),
        angular_velocity=physics.angular_velocity.clone(),
        energy=physics.energy.clone(),
        alive=physics.alive.clone(),
    )

    return CollisionResolution(
        physics=result_physics,
        contact=contact.reshape(batch_shape + (1,)),
        unresolved_contact=unresolved.reshape(batch_shape + (1,)),
        boundary_escape=boundary_escape.reshape(batch_shape + (1,)),
    )
