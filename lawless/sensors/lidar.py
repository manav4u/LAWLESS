"""Eight-ray, body-relative lidar sensing for LAWLESS worlds.

Precision policy:
    Agent and obstacle geometry must use torch.float32 or torch.float64.
    All geometric calculations are promoted to torch.float64 on the input
    device. This improves arithmetic stability but cannot recover precision
    already lost when input values were stored in float32.

    Ray directions are formed by rotating an exact body-relative direction
    table by the stored heading. No angular dead zone or radius tolerance is
    applied. Circle tangency is inclusive under the computed float64
    perpendicular-distance comparison.

The sensor is read-only: it does not mutate agent, world, or obstacle state.
"""

from __future__ import annotations

import math

import torch

from lawless.physics.state import AgentState
from lawless.world.config import SensorConfig
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState


_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_RAY_COUNT = 8


def _validate_world_geometry(world: WorldState) -> None:
    """Revalidate mutable tensor contents at the sensing boundary."""
    if not isinstance(world, WorldState):
        raise TypeError("world must be a WorldState")

    for name in ("width_m", "height_m"):
        value = getattr(world, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"world.{name} must be finite and positive")

    obstacles = world.obstacles
    if not isinstance(obstacles, CircleObstacles):
        raise TypeError("world.obstacles must be CircleObstacles")

    centers = obstacles.centers
    radii = obstacles.radii

    if not isinstance(centers, torch.Tensor):
        raise TypeError("world.obstacles.centers must be a Tensor")
    if not isinstance(radii, torch.Tensor):
        raise TypeError("world.obstacles.radii must be a Tensor")

    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError("world.obstacles.centers must have shape [M, 2]")
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise ValueError("world.obstacles.radii must have shape [M]")

    if centers.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("obstacle geometry must use torch.float32 or torch.float64")
    if radii.dtype != centers.dtype:
        raise TypeError("obstacle centers and radii must have the same dtype")
    if radii.device != centers.device:
        raise ValueError("obstacle centers and radii must share a device")

    if not bool(torch.isfinite(centers).all().item()):
        raise ValueError("obstacle centers must be finite")
    if not bool(torch.isfinite(radii).all().item()):
        raise ValueError("obstacle radii must be finite")
    if not bool((radii > 0).all().item()):
        raise ValueError("obstacle radii must be strictly positive")

    # Recheck containment because frozen dataclasses do not make tensor
    # contents immutable.
    if centers.shape[0]:
        width = torch.as_tensor(
            world.width_m, dtype=centers.dtype, device=centers.device
        )
        height = torch.as_tensor(
            world.height_m, dtype=centers.dtype, device=centers.device
        )
        contained = (
            (centers[:, 0] - radii >= 0)
            & (centers[:, 1] - radii >= 0)
            & (centers[:, 0] + radii <= width)
            & (centers[:, 1] + radii <= height)
        )
        if not bool(contained.all().item()):
            raise ValueError("every obstacle must be fully contained in the world")


def _validate_inputs(
    state: AgentState,
    world: WorldState,
    sensor_config: SensorConfig,
) -> None:
    if not isinstance(state, AgentState):
        raise TypeError("state must be an AgentState")
    if not isinstance(sensor_config, SensorConfig):
        raise TypeError("sensor_config must be a SensorConfig")

    sensor_config.validate()
    _validate_world_geometry(world)

    fields = (
        ("position", state.position),
        ("velocity", state.velocity),
        ("angle", state.angle),
        ("angular_velocity", state.angular_velocity),
        ("energy", state.energy),
        ("alive", state.alive),
    )
    for name, value in fields:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state.{name} must be a Tensor")

    position = state.position
    if position.ndim < 1 or position.shape[-1] != 2:
        raise ValueError("state.position must have shape [..., 2]")
    if position.dtype not in _SUPPORTED_DTYPES:
        raise TypeError("agent geometry must use torch.float32 or torch.float64")
    if not position.is_floating_point():
        raise TypeError("state.position must be floating point")

    if state.alive.dtype != torch.bool:
        raise TypeError("state.alive must have dtype torch.bool")

    expected_shapes = {
        "velocity": position.shape,
        "angle": position.shape[:-1] + (1,),
        "angular_velocity": position.shape[:-1] + (1,),
        "energy": position.shape[:-1] + (1,),
        "alive": position.shape[:-1] + (1,),
    }
    for name, value in fields[1:]:
        if value.shape != expected_shapes[name]:
            raise ValueError(f"state.{name} has an invalid shape")

    for name, value in fields[1:5]:
        if value.dtype != position.dtype:
            raise TypeError(f"state.{name} must have the same dtype as position")
        if value.device != position.device:
            raise ValueError(f"state.{name} must share the position device")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"state.{name} must be finite")

    if state.alive.device != position.device:
        raise ValueError("state.alive must share the position device")

    if not bool(torch.isfinite(position).all().item()):
        raise ValueError("state.position must be finite")

    # Keep the accepted AgentState contract as the authoritative shape check.
    state.validate_shapes()

    obstacles = world.obstacles
    if obstacles.centers.device != position.device:
        raise ValueError("agent and obstacle geometry must share a device")
    if obstacles.centers.dtype != position.dtype:
        raise TypeError("agent and obstacle geometry must share a dtype")


def _ray_directions(angles: torch.Tensor) -> torch.Tensor:
    """Return [..., 8, 2] unit directions in world coordinates.

    ``angles`` has shape [..., 1]. The exact base table prevents trigonometric
    residuals for zero-heading cardinal rays. A nonzero heading is always
    applied, however small it is.
    """
    theta = angles.to(dtype=torch.float64).squeeze(-1)

    half_diagonal = math.sqrt(0.5)
    base = torch.tensor(
        [
            [1.0, 0.0],
            [half_diagonal, half_diagonal],
            [0.0, 1.0],
            [-half_diagonal, half_diagonal],
            [-1.0, 0.0],
            [-half_diagonal, -half_diagonal],
            [0.0, -1.0],
            [half_diagonal, -half_diagonal],
        ],
        dtype=torch.float64,
        device=angles.device,
    )

    cos_theta = torch.cos(theta).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1)

    base_x = base[:, 0]
    base_y = base[:, 1]

    dx = cos_theta * base_x - sin_theta * base_y
    dy = sin_theta * base_x + cos_theta * base_y

    directions = torch.stack((dx, dy), dim=-1)

    # Normalize after rotation so projection and perpendicular distance use
    # a unit direction despite finite-precision trigonometric evaluation.
    norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    return directions / norms


def _ray_circle_distances(
    origins: torch.Tensor,
    directions: torch.Tensor,
    obstacles: CircleObstacles,
) -> torch.Tensor:
    """Nearest nonnegative ray/circle surface distance, or +inf.

    Inputs are promoted to float64 by the caller.

    Exact surface contact is handled independently of ray direction:
    if the represented origin has squared center distance exactly equal
    to the represented radius squared, the distance is zero for every ray.

    No epsilon, radius inflation, or near-contact tolerance is used.
    Strictly interior origins use the forward exit surface. Strictly
    exterior origins use the first forward intersection, if one exists.
    """
    if obstacles.centers.shape[0] == 0:
        return torch.full(
            origins.shape[:-1],
            float("inf"),
            dtype=torch.float64,
            device=origins.device,
        )

    centers = obstacles.centers.to(dtype=torch.float64)
    radii = obstacles.radii.to(dtype=torch.float64)
    relative = origins.unsqueeze(-2) - centers

    # Exact represented-geometry contact test. This is deliberately
    # independent of ray direction and introduces no epsilon halo.
    origin_distance_squared = relative.square().sum(dim=-1)
    radius_squared = radii.square()
    exact_surface_contact = origin_distance_squared == radius_squared

    dx = directions[..., 0].unsqueeze(-1)
    dy = directions[..., 1].unsqueeze(-1)
    rx = relative[..., 0]
    ry = relative[..., 1]

    direction_norm = torch.sqrt(dx.square() + dy.square())
    projection = (rx * dx + ry * dy) / direction_norm
    perpendicular = (rx * dy - ry * dx).abs() / direction_norm

    intersects = perpendicular <= radii
    half_chord_squared = (
        (radii - perpendicular).clamp_min(0.0)
        * (radii + perpendicular)
    )
    half_chord = torch.sqrt(half_chord_squared)

    near = -projection - half_chord
    far = -projection + half_chord

    # For strictly interior origins, near is behind the ray and far is
    # the forward exit. For exterior origins, near is the first hit.
    first_forward = torch.where(near >= 0.0, near, far)
    valid_intersection = intersects & (first_forward >= 0.0)

    # Exact contact overrides directional intersection arithmetic.
    distances = torch.where(
        exact_surface_contact,
        torch.zeros_like(first_forward),
        torch.where(
            valid_intersection,
            first_forward,
            torch.full_like(first_forward, float("inf")),
        ),
    )

    return distances.amin(dim=-1)


def _ray_boundary_distances(
    origins: torch.Tensor,
    directions: torch.Tensor,
    width: float,
    height: float,
) -> torch.Tensor:
    """Distance to the first forward exit from the closed world rectangle."""
    x = origins[..., 0]
    y = origins[..., 1]
    dx = directions[..., 0]
    dy = directions[..., 1]

    width_t = torch.as_tensor(width, dtype=torch.float64, device=origins.device)
    height_t = torch.as_tensor(height, dtype=torch.float64, device=origins.device)

    inf_x = torch.full_like(dx, float("inf"))
    inf_y = torch.full_like(dy, float("inf"))

    tx = torch.where(
        dx > 0.0,
        (width_t - x) / dx,
        torch.where(dx < 0.0, (0.0 - x) / dx, inf_x),
    )
    ty = torch.where(
        dy > 0.0,
        (height_t - y) / dy,
        torch.where(dy < 0.0, (0.0 - y) / dy, inf_y),
    )

    tx = torch.where(tx >= 0.0, tx, inf_x)
    ty = torch.where(ty >= 0.0, ty, inf_y)
    return torch.minimum(tx, ty)


def sense_lidar(
    state: AgentState,
    world: WorldState,
    config: SensorConfig,
) -> torch.Tensor:
    """Return eight normalized body-relative lidar distances per agent.

    The final dimension is ordered at body-relative offsets
    ``2*pi*i/8``, for ``i=0..7``. Values are in [0, 1], with 1 representing
    no hit within range. Dead agents receive all ones, including when their
    stored positions are outside the world.
    """
    _validate_inputs(state, world, config)

    position = state.position
    alive = state.alive.squeeze(-1)

    # Both operands have exactly the batch shape: [...]. Dead agents are
    # excluded individually and may therefore have positions outside the
    # world without affecting live agents elsewhere in the batch.
    outside_world = ~world.contains_positions(position)
    live_outside_world = alive & outside_world
    if bool(live_outside_world.any().item()):
        raise ValueError("live agent positions must be inside the world")

    # Compute geometry in float64 while retaining the original device.
    origins = position.to(dtype=torch.float64)
    directions = _ray_directions(state.angle)

    origins_per_ray = origins.unsqueeze(-2).expand(
        origins.shape[:-1] + (_RAY_COUNT, 2)
    )

    obstacle_distances = _ray_circle_distances(
        origins_per_ray,
        directions,
        world.obstacles,
    )
    boundary_distances = _ray_boundary_distances(
        origins_per_ray,
        directions,
        world.width_m,
        world.height_m,
    )

    nearest = torch.minimum(obstacle_distances, boundary_distances)
    max_range = float(config.lidar_range_m)

    normalized = (nearest / max_range).clamp(min=0.0, max=1.0)
    normalized = torch.where(
        torch.isfinite(nearest),
        normalized,
        torch.ones_like(normalized),
    )

    normalized = torch.where(
        alive.unsqueeze(-1),
        normalized,
        torch.ones_like(normalized),
    )

    return normalized.to(dtype=position.dtype, device=position.device)
