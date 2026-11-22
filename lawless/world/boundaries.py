"""Axis-aligned rectangular world boundary queries.

Coordinates are in metres. The world is the closed rectangle
[0, width_m] x [0, height_m]. Points on edges and corners are inside.
Only points strictly beyond an edge are outside.

This module classifies positions; it does not clamp, reflect, wrap,
or otherwise modify them.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def _validate_world_size(width_m: float, height_m: float) -> None:
    for name, value in (("width_m", width_m), ("height_m", height_m)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and strictly positive")


def _validate_positions(positions: Tensor) -> None:
    if not isinstance(positions, Tensor):
        raise TypeError("positions must be a torch.Tensor")
    if positions.ndim < 1 or positions.shape[-1] != 2:
        raise ValueError("positions must have shape [..., 2]")
    if not positions.is_floating_point():
        raise TypeError("positions must have a floating-point dtype")
    if not bool(torch.isfinite(positions).all()):
        raise ValueError("positions must contain only finite values")


def inside_world(
    positions: Tensor,
    width_m: float = 100.0,
    height_m: float = 100.0,
) -> Tensor:
    """Return whether each position is inside or on the world boundary.

    Input:  [..., 2]
    Output: [...] boolean tensor on the input device.
    """
    _validate_positions(positions)
    _validate_world_size(width_m, height_m)

    x = positions[..., 0]
    y = positions[..., 1]
    return (
        (x >= 0)
        & (x <= width_m)
        & (y >= 0)
        & (y <= height_m)
    )


def outside_world(
    positions: Tensor,
    width_m: float = 100.0,
    height_m: float = 100.0,
) -> Tensor:
    """Return the logical complement of inside_world."""
    return ~inside_world(positions, width_m, height_m)


def boundary_clearance(
    positions: Tensor,
    width_m: float = 100.0,
    height_m: float = 100.0,
) -> Tensor:
    """Return minimum signed distance to the four world edges.

    Input:  [..., 2]
    Output: [...] floating-point tensor, preserving input dtype/device.

    Returns min(x, width_m - x, y, height_m - y).
    Positive means inside, zero means on an edge, and negative means
    outside. For an interior point, this is the distance to its nearest
    edge. For an exterior point, it is the negative of the largest
    axis-aligned boundary violation. It is not Euclidean distance to
    the rectangle, particularly outside corners.
    """
    _validate_positions(positions)
    _validate_world_size(width_m, height_m)

    x = positions[..., 0]
    y = positions[..., 1]

    distances = torch.stack(
        (
            x,
            width_m - x,
            y,
            height_m - y,
        ),
        dim=-1,
    )
    return distances.amin(dim=-1)
