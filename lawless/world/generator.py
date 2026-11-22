"""Deterministic, obstacle-only world generation.

Random candidates are sampled on CPU with a private torch.Generator.
Only float32 and float64 are supported. Both output dtypes are checked
against placement invariants after generation and device transfer.

Configured obstacle-radius bounds are converted to the requested output
dtype before sampling. Thus, for example, a configured fixed radius of 0.1
in float32 means the nearest representable float32 value to 0.1, not exact
real-number equality with 0.1.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch

from lawless.world.config import WorldConfig
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState


class WorldGenerationError(RuntimeError):
    """Raised when a valid obstacle layout cannot be generated within its limit."""


_SUPPORTED_DTYPES = (torch.float32, torch.float64)
_MAX_SEED = (1 << 63) - 1


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite real number")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_inputs(
    config: WorldConfig,
    seed: int,
    max_attempts_per_obstacle: int,
    dtype: torch.dtype,
) -> tuple[float, float, int, float, float]:
    if not isinstance(config, WorldConfig):
        raise TypeError("config must be a WorldConfig")

    # The canonical config validator owns configuration semantics.
    # Do not duplicate or weaken its validation contract here.
    config.validate()

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer (bool is not accepted)")
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError(f"seed must be in [0, {_MAX_SEED}]")

    if (
        isinstance(max_attempts_per_obstacle, bool)
        or not isinstance(max_attempts_per_obstacle, Integral)
    ):
        raise TypeError("max_attempts_per_obstacle must be an integer")
    if max_attempts_per_obstacle <= 0:
        raise ValueError("max_attempts_per_obstacle must be positive")

    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "dtype must be torch.float32 or torch.float64; "
            f"received {dtype}"
        )

    width = _finite_number("width_m", config.width_m)
    height = _finite_number("height_m", config.height_m)
    configured_radius_min = _finite_number(
        "obstacle_radius_min_m", config.obstacle_radius_min_m
    )
    configured_radius_max = _finite_number(
        "obstacle_radius_max_m", config.obstacle_radius_max_m
    )

    count = config.obstacle_count

    # Interpret configured bounds in the requested output dtype. This avoids
    # claiming exact real-number representation for decimal values such as
    # 0.1 in float32.
    radius_bounds = torch.tensor(
        [configured_radius_min, configured_radius_max],
        dtype=dtype,
        device="cpu",
    )

    if not bool(torch.isfinite(radius_bounds).all()):
        raise ValueError(
            "configured obstacle radius bounds must be finite in the output dtype"
        )

    radius_min = float(radius_bounds[0].item())
    radius_max = float(radius_bounds[1].item())

    if radius_min <= 0.0 or radius_max <= 0.0:
        raise ValueError(
            "configured obstacle radius bounds must remain positive in the output dtype"
        )
    if radius_min > radius_max:
        raise ValueError(
            "configured obstacle radius bounds are reversed in the output dtype"
        )

    # Radius fit is a packing constraint only when obstacles are requested.
    # An empty world does not use the configured obstacle radii.
    if count > 0 and 2.0 * radius_max > min(width, height):
        raise ValueError(
            "maximum obstacle diameter must fit inside both world dimensions"
        )

    return width, height, int(count), radius_min, radius_max


def _validate_final_geometry(
    centers: torch.Tensor,
    radii: torch.Tensor,
    width: float,
    height: float,
    radius_min: float,
    radius_max: float,
) -> None:
    """Validate geometry using the exact values returned to the caller."""
    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise RuntimeError("generated obstacle centers have an invalid shape")
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise RuntimeError("generated obstacle radii have an invalid shape")
    if centers.dtype != radii.dtype or centers.device != radii.device:
        raise RuntimeError("generated obstacle tensors differ in dtype or device")

    if not bool(torch.isfinite(centers).all()):
        raise RuntimeError("generated obstacle centers are not finite")
    if not bool(torch.isfinite(radii).all()):
        raise RuntimeError("generated obstacle radii are not finite")
    if not bool((radii > 0).all()):
        raise RuntimeError("generated obstacle radii must be positive")

    # These bounds are the configured values rounded to the requested output
    # dtype. Comparisons are therefore made against representable values,
    # rather than the original Python decimal values.
    within_configured_bounds = (radii >= radius_min) & (radii <= radius_max)
    if not bool(within_configured_bounds.all()):
        raise RuntimeError(
            "generated obstacle radii violate configured output-dtype bounds"
        )

    if centers.shape[0] == 0:
        return

    x, y = centers.unbind(dim=-1)
    contained = (
        (x - radii >= 0)
        & (x + radii <= width)
        & (y - radii >= 0)
        & (y + radii <= height)
    )
    if not bool(contained.all()):
        raise RuntimeError("generated obstacle is not fully inside the world")

    if centers.shape[0] > 1:
        delta = centers[:, None, :] - centers[None, :, :]
        distances = torch.linalg.vector_norm(delta, dim=-1)
        summed_radii = radii[:, None] + radii[None, :]
        upper = torch.triu(
            torch.ones_like(distances, dtype=torch.bool),
            diagonal=1,
        )

        # Strictly positive clearance: overlap and tangency are both rejected.
        if not bool((distances[upper] > summed_radii[upper]).all()):
            raise RuntimeError(
                "generated obstacles overlap or touch in the returned dtype"
            )


def generate_obstacle_world(
    config: WorldConfig,
    *,
    seed: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    max_attempts_per_obstacle: int = 10_000,
) -> WorldState:
    """Generate one seeded world containing only non-overlapping obstacles.

    Args:
        config: A canonical ``WorldConfig``. Its existing ``validate()``
            contract is authoritative. The config is not mutated.
        seed: Explicit nonnegative integer application seed in the inclusive
            range [0, 2**63 - 1]. This is LAWLESS's seed policy, not a claim
            about the full range accepted by torch.Generator.
        device: Destination device, such as ``"cpu"`` or ``"cuda"``.
        dtype: ``torch.float32`` or ``torch.float64`` only. Configured
            obstacle-radius bounds are first rounded to this dtype; generated
            radii must lie within those representable bounds.
        max_attempts_per_obstacle: Maximum candidate placements per obstacle.

    Returns:
        A validated single-world ``WorldState``. Zero obstacles are supported.

    Raises:
        TypeError or ValueError: Invalid arguments or configuration.
        WorldGenerationError: A requested obstacle could not be placed within
            its finite attempt budget. No partial world is returned.

    Reproducibility:
        Random draws are made on CPU using a private ``torch.Generator``.
        The same seed, config, dtype, and PyTorch generator implementation
        produce the same CPU-sampled layout. Transferring the generated
        tensors to CPU or CUDA preserves their values exactly for the
        supported dtypes. Bitwise reproducibility across different PyTorch
        versions is not promised.
    """
    width, height, count, radius_min, radius_max = _validate_inputs(
        config,
        seed,
        max_attempts_per_obstacle,
        dtype,
    )

    try:
        destination = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc

    # All random draws occur on CPU. No global RNG state is read or changed.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    centers_cpu = torch.empty((count, 2), dtype=dtype, device="cpu")
    radii_cpu = torch.empty((count,), dtype=dtype, device="cpu")

    for obstacle_index in range(count):
        placed = False

        for _attempt in range(int(max_attempts_per_obstacle)):
            u_radius = torch.rand(
                (),
                generator=generator,
                dtype=dtype,
                device="cpu",
            )
            radius = radius_min + (radius_max - radius_min) * u_radius

            # Place the full circle inside the closed world rectangle.
            x = radius + torch.rand(
                (),
                generator=generator,
                dtype=dtype,
                device="cpu",
            ) * (width - 2.0 * radius)

            y = radius + torch.rand(
                (),
                generator=generator,
                dtype=dtype,
                device="cpu",
            ) * (height - 2.0 * radius)

            candidate = torch.stack((x, y))

            if obstacle_index > 0:
                delta = centers_cpu[:obstacle_index] - candidate
                distances = torch.linalg.vector_norm(delta, dim=-1)
                clearances = distances - (
                    radii_cpu[:obstacle_index] + radius
                )

                # Strictly positive clearance; tangency is rejected.
                if not bool((clearances > 0).all()):
                    continue

            centers_cpu[obstacle_index] = candidate
            radii_cpu[obstacle_index] = radius
            placed = True
            break

        if not placed:
            raise WorldGenerationError(
                "obstacle placement exhausted its candidate budget: "
                f"category='obstacle', index={obstacle_index}, "
                f"max_attempts={max_attempts_per_obstacle}, seed={seed}. "
                "No partial world was returned."
            )

    # Transfer only after the complete CPU layout has been generated.
    centers = centers_cpu.to(device=destination)
    radii = radii_cpu.to(device=destination)

    # Revalidate the exact returned geometry and dtype-rounded radius bounds.
    _validate_final_geometry(
        centers,
        radii,
        width,
        height,
        radius_min,
        radius_max,
    )

    obstacles = CircleObstacles(centers=centers, radii=radii)
    return WorldState.from_config(config, obstacles)
