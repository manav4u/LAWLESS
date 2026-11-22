"""Full initial-layout generation for the LAWLESS P1 environment.

This module composes the existing obstacle-only generator with spawn,
extraction, and core placement. All sampling uses one private CPU generator.
The obstacle helper runs first and is not modified.

The returned GeneratedWorld contains geometry only. Mutable episode state,
including per-agent core-collection ownership, belongs to later stages.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch

from lawless.world.config import EnvironmentConfig
from lawless.world.generated_world import GeneratedWorld
from lawless.world.generator import (
    WorldGenerationError,
    _generate_obstacles_with_rng,
    _validate_inputs,
)
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState


_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_MAX_SEED = (1 << 63) - 1


def _positive_dtype_scalar(
    name: str,
    value: object,
    dtype: torch.dtype,
) -> float:
    """Validate a positive scalar and return its dtype-rounded value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive real number")

    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")

    rounded = torch.tensor(numeric, dtype=dtype, device="cpu")
    if not bool(torch.isfinite(rounded)) or not bool(rounded > 0):
        raise ValueError(
            f"{name} must remain finite and positive in the layout dtype"
        )
    return float(rounded.item())


def _finite_dimension(
    name: str,
    value: object,
    dtype: torch.dtype,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite positive real number")

    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")

    represented = torch.tensor(numeric, dtype=dtype, device="cpu")
    if not bool(torch.isfinite(represented)) or not bool(represented > 0):
        raise ValueError(
            f"{name} must remain finite and positive in the layout dtype"
        )
    return numeric


def _validate_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer (bool is not accepted)")
    result = int(seed)
    if not 0 <= result <= _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}]")
    return result


def _validate_count(name: str, count: object) -> int:
    if isinstance(count, bool) or not isinstance(count, Integral):
        raise TypeError(f"{name} must be an integer (bool is not accepted)")
    result = int(count)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _validate_budget(name: str, budget: object) -> int:
    if isinstance(budget, bool) or not isinstance(budget, Integral):
        raise TypeError(f"{name} must be an integer (bool is not accepted)")
    result = int(budget)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_device(device: torch.device | str) -> torch.device:
    try:
        destination = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc

    if destination.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if destination.type not in ("cpu", "cuda"):
        raise ValueError("device must be a CPU or CUDA device")
    if destination.type == "cuda" and destination.index is not None:
        if destination.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is unavailable: {destination}")
    return destination


def _candidate_center(
    *,
    width: float,
    height: float,
    radius: float,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Draw one full-circle-contained center using two scalar CPU draws."""
    u_x = torch.rand((), generator=generator, dtype=dtype, device="cpu")
    x = radius + u_x * (width - 2.0 * radius)

    u_y = torch.rand((), generator=generator, dtype=dtype, device="cpu")
    y = radius + u_y * (height - 2.0 * radius)

    return torch.stack((x, y))


def _has_strict_clearance(
    candidate_center: torch.Tensor,
    candidate_radius: float,
    other_centers: torch.Tensor,
    other_radii: torch.Tensor,
) -> bool:
    if other_centers.shape[0] == 0:
        return True

    distances = torch.linalg.vector_norm(
        other_centers - candidate_center,
        dim=-1,
    )
    return bool((distances > other_radii + candidate_radius).all())


def _inside_world(
    center: torch.Tensor,
    radius: float,
    width: float,
    height: float,
) -> bool:
    x, y = center.unbind(dim=-1)
    return bool(
        (x - radius >= 0)
        & (x + radius <= width)
        & (y - radius >= 0)
        & (y + radius <= height)
    )


def _place_circle_category(
    *,
    category: str,
    count: int,
    radius: float,
    width: float,
    height: float,
    generator: torch.Generator,
    dtype: torch.dtype,
    max_attempts: int,
    seed: int,
    fixed_centers: tuple[torch.Tensor, ...],
    fixed_radii: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Place a sequence of equal-radius circles with bounded rejection."""
    centers: list[torch.Tensor] = []

    for index in range(count):
        placed = False

        for _attempt in range(max_attempts):
            candidate = _candidate_center(
                width=width,
                height=height,
                radius=radius,
                generator=generator,
                dtype=dtype,
            )

            if not _inside_world(candidate, radius, width, height):
                continue

            clear = True

            for other_centers, other_radii in zip(
                fixed_centers, fixed_radii
            ):
                if not _has_strict_clearance(
                    candidate,
                    radius,
                    other_centers,
                    other_radii,
                ):
                    clear = False
                    break

            if clear and centers:
                prior_centers = torch.stack(centers)
                prior_radii = torch.full(
                    (len(centers),),
                    radius,
                    dtype=dtype,
                    device="cpu",
                )
                clear = _has_strict_clearance(
                    candidate,
                    radius,
                    prior_centers,
                    prior_radii,
                )

            if clear:
                centers.append(candidate)
                placed = True
                break

        if not placed:
            raise WorldGenerationError(
                f"{category} placement exhausted its candidate budget: "
                f"category={category!r}, index={index}, "
                f"max_attempts={max_attempts}, seed={seed}. "
                "No partial layout was returned; exhaustion does not "
                "establish mathematical infeasibility."
            )

    if centers:
        center_tensor = torch.stack(centers)
    else:
        center_tensor = torch.empty((0, 2), dtype=dtype, device="cpu")

    radius_tensor = torch.full(
        (count,),
        radius,
        dtype=dtype,
        device="cpu",
    )
    return center_tensor, radius_tensor


def _place_single_circle(
    *,
    category: str,
    radius: float,
    width: float,
    height: float,
    generator: torch.Generator,
    dtype: torch.dtype,
    max_attempts: int,
    seed: int,
    other_centers: tuple[torch.Tensor, ...],
    other_radii: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    for _attempt in range(max_attempts):
        candidate = _candidate_center(
            width=width,
            height=height,
            radius=radius,
            generator=generator,
            dtype=dtype,
        )

        if not _inside_world(candidate, radius, width, height):
            continue

        if all(
            _has_strict_clearance(
                candidate,
                radius,
                centers,
                radii,
            )
            for centers, radii in zip(other_centers, other_radii)
        ):
            return candidate

    raise WorldGenerationError(
        f"{category} placement exhausted its candidate budget: "
        f"category={category!r}, index=0, "
        f"max_attempts={max_attempts}, seed={seed}. "
        "No partial layout was returned; exhaustion does not establish "
        "mathematical infeasibility."
    )


def generate_world(
    config: EnvironmentConfig,
    *,
    seed: int | None = None,
    spawn_count: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    max_attempts_per_obstacle: int = 10_000,
    max_attempts_per_spawn: int = 10_000,
    max_attempts_per_extraction: int = 10_000,
    max_attempts_per_core: int = 10_000,
) -> GeneratedWorld:
    """Generate one complete, validated initial world layout.

    Placement order and random stream:
      1. Obstacles, using the existing obstacle-only helper unchanged.
      2. Spawn slots, in ascending slot index.
      3. Extraction circle.
      4. Cores, in ascending core index.

    One private CPU torch.Generator is used for the entire operation.
    The obstacle helper consumes the first random draws, preserving the
    historical obstacle-only generator's sampling prefix.

    Args:
        config: Validated LAWLESS EnvironmentConfig.
        seed: Optional explicit application seed. None uses config.seed.
        spawn_count: Number of distinct simultaneous spawn slots.
        device: Final tensor device, CPU or CUDA.
        dtype: torch.float32 or torch.float64.
        max_attempts_per_obstacle: Candidate budget for each obstacle.
        max_attempts_per_spawn: Candidate budget for each spawn slot.
        max_attempts_per_extraction: Candidate budget for extraction.
        max_attempts_per_core: Candidate budget for each core.

    Returns:
        A GeneratedWorld containing shared static geometry and initial
        spawn slots. It contains no episode state or collection masks.

    Raises:
        TypeError / ValueError: Invalid configuration or arguments.
        WorldGenerationError: A requested category exhausts its finite
            candidate budget. No partial layout is returned.
    """
    if not isinstance(config, EnvironmentConfig):
        raise TypeError("config must be an EnvironmentConfig")

    # Validate the complete configuration before creating or consuming RNG.
    config.validate()

    effective_seed = _validate_seed(config.seed if seed is None else seed)
    spawn_count = _validate_count("spawn_count", spawn_count)

    obstacle_budget = _validate_budget(
        "max_attempts_per_obstacle", max_attempts_per_obstacle
    )
    spawn_budget = _validate_budget(
        "max_attempts_per_spawn", max_attempts_per_spawn
    )
    extraction_budget = _validate_budget(
        "max_attempts_per_extraction", max_attempts_per_extraction
    )
    core_budget = _validate_budget(
        "max_attempts_per_core", max_attempts_per_core
    )

    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError("dtype must be torch.float32 or torch.float64")

    destination = _validate_device(device)

    world_config = config.world
    width = _finite_dimension("world.width_m", world_config.width_m, dtype)
    height = _finite_dimension("world.height_m", world_config.height_m, dtype)

    # Validate the obstacle configuration and round its radius bounds exactly
    # as the historical obstacle-only public generator does.
    (
        _obstacle_width,
        _obstacle_height,
        obstacle_count,
        obstacle_radius_min,
        obstacle_radius_max,
    ) = _validate_inputs(
        world_config,
        effective_seed,
        obstacle_budget,
        dtype,
    )

    agent_radius = _positive_dtype_scalar(
        "embodiment.agent_radius_m",
        config.embodiment.agent_radius_m,
        dtype,
    )
    extraction_radius = _positive_dtype_scalar(
        "objectives.extraction_radius_m",
        config.objectives.extraction_radius_m,
        dtype,
    )
    core_radius = _positive_dtype_scalar(
        "objectives.core_radius_m",
        config.objectives.core_radius_m,
        dtype,
    )
    core_count = _validate_count(
        "objectives.core_count",
        config.objectives.core_count,
    )

    # Reject impossible single-circle containment as invalid geometry before
    # any RNG draw. Packing interactions remain bounded placement outcomes.
    for name, radius in (
        ("obstacle", obstacle_radius_max),
        ("spawn body", agent_radius),
        ("extraction", extraction_radius),
        ("core", core_radius),
    ):
        if 2.0 * radius > min(width, height):
            if name == "obstacle" and obstacle_count == 0:
                continue
            if name == "spawn body" and spawn_count == 0:
                continue
            if name == "core" and core_count == 0:
                continue
            raise ValueError(
                f"maximum {name} diameter must fit inside both "
                "world dimensions"
            )

    # The single RNG is seeded once. Do not move this below obstacle
    # generation: the obstacle helper must consume the stream prefix.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(effective_seed)

    obstacles_cpu = _generate_obstacles_with_rng(
        width=width,
        height=height,
        count=obstacle_count,
        radius_min=obstacle_radius_min,
        radius_max=obstacle_radius_max,
        seed=effective_seed,
        generator=generator,
        device="cpu",
        dtype=dtype,
        max_attempts_per_obstacle=obstacle_budget,
    )

    obstacle_centers = obstacles_cpu.centers
    obstacle_radii = obstacles_cpu.radii

    spawn_centers, spawn_radii = _place_circle_category(
        category="spawn",
        count=spawn_count,
        radius=agent_radius,
        width=width,
        height=height,
        generator=generator,
        dtype=dtype,
        max_attempts=spawn_budget,
        seed=effective_seed,
        fixed_centers=(obstacle_centers,),
        fixed_radii=(obstacle_radii,),
    )

    extraction_center_cpu = _place_single_circle(
        category="extraction",
        radius=extraction_radius,
        width=width,
        height=height,
        generator=generator,
        dtype=dtype,
        max_attempts=extraction_budget,
        seed=effective_seed,
        other_centers=(obstacle_centers, spawn_centers),
        other_radii=(
            obstacle_radii,
            torch.full(
                (spawn_count,),
                agent_radius,
                dtype=dtype,
                device="cpu",
            ),
        ),
    )

    core_centers, core_radii = _place_circle_category(
        category="core",
        count=core_count,
        radius=core_radius,
        width=width,
        height=height,
        generator=generator,
        dtype=dtype,
        max_attempts=core_budget,
        seed=effective_seed,
        fixed_centers=(
            obstacle_centers,
            spawn_centers,
            extraction_center_cpu.reshape(1, 2),
        ),
        fixed_radii=(
            obstacle_radii,
            torch.full(
                (spawn_count,),
                agent_radius,
                dtype=dtype,
                device="cpu",
            ),
            torch.tensor(
                [extraction_radius],
                dtype=dtype,
                device="cpu",
            ),
        ),
    )

    # Transfer only after all categories have been successfully generated.
    obstacles = CircleObstacles(
        centers=obstacle_centers.to(device=destination),
        radii=obstacle_radii.to(device=destination),
    )
    world = WorldState.from_config(world_config, obstacles)

    result = GeneratedWorld(
        world=world,
        spawn_positions=spawn_centers.to(device=destination),
        agent_body_radius_m=agent_radius,
        extraction_center=extraction_center_cpu.to(device=destination),
        extraction_radius_m=extraction_radius,
        core_positions=core_centers.to(device=destination),
        core_radii=core_radii.to(device=destination),
        seed=effective_seed,
    )

    return result