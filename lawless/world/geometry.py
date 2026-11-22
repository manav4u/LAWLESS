"""Vectorized 2-D circle geometry.

All distances and coordinates are expressed in metres.
Contact is inclusive: exact tangency counts as contact.
This module detects geometric relationships only; it performs no
collision response or state mutation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _require_points(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < 1 or value.shape[-1] != 2:
        raise ValueError(f"{name} must have shape [..., 2]")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class CircleObstacles:
    """A batch of static circles.

    centers: [M, 2] floating-point tensor, in metres.
    radii:   [M] floating-point tensor, in metres.

    An empty set is represented by centers.shape == (0, 2) and
    radii.shape == (0,). The tensors must share dtype and device.
    """

    centers: Tensor
    radii: Tensor

    def __post_init__(self) -> None:
        centers = self.centers
        radii = self.radii

        if not isinstance(centers, Tensor) or not isinstance(radii, Tensor):
            raise TypeError("centers and radii must be torch.Tensor objects")
        if centers.ndim != 2 or centers.shape[-1] != 2:
            raise ValueError("centers must have shape [M, 2]")
        if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
            raise ValueError("radii must have shape [M], matching centers")
        if not centers.is_floating_point() or not radii.is_floating_point():
            raise TypeError("centers and radii must have floating-point dtypes")
        if centers.dtype != radii.dtype:
            raise TypeError("centers and radii must have the same dtype")
        if centers.device != radii.device:
            raise ValueError("centers and radii must be on the same device")
        if not bool(torch.isfinite(centers).all()):
            raise ValueError("centers must contain only finite values")
        if not bool(torch.isfinite(radii).all()):
            raise ValueError("radii must contain only finite values")
        if bool((radii <= 0).any()):
            raise ValueError("circle radii must be strictly positive")

    @property
    def count(self) -> int:
        return self.centers.shape[0]

    @classmethod
    def empty(
        cls,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "CircleObstacles":
        """Create a valid zero-obstacle set."""
        if not dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch dtype")
        return cls(
            centers=torch.empty((0, 2), device=device, dtype=dtype),
            radii=torch.empty((0,), device=device, dtype=dtype),
        )


def point_circle_signed_clearance(
    points: Tensor,
    obstacles: CircleObstacles,
) -> Tensor:
    """Return signed point-to-circle clearance with shape [..., M].

    Positive: outside the circle.
    Zero:     exactly on its circumference.
    Negative: inside the circle.

    This is center distance minus circle radius. For zero obstacles,
    the result has shape [..., 0].
    """
    _require_points("points", points)

    if points.device != obstacles.centers.device:
        raise ValueError("points and obstacles must be on the same device")
    if points.dtype != obstacles.centers.dtype:
        raise ValueError("points and obstacles must have the same dtype")

    distances = torch.linalg.vector_norm(
        points.unsqueeze(-2) - obstacles.centers,
        dim=-1,
    )
    return distances - obstacles.radii


def point_circle_contact(
    points: Tensor,
    obstacles: CircleObstacles,
) -> Tensor:
    """Return inclusive point-circle contact flags with shape [..., M].

    Exact tangency is contact. With zero obstacles, returns a boolean
    tensor with shape [..., 0].
    """
    return point_circle_signed_clearance(points, obstacles) <= 0


def circle_circle_signed_clearance(
    centers_a: Tensor,
    radii_a: Tensor,
    centers_b: Tensor,
    radii_b: Tensor,
) -> Tensor:
    """Return pairwise circle clearance with shape [A, B].

    Clearance is center distance minus the sum of the two radii.
    Inputs are centers [A, 2], radii [A], centers [B, 2], radii [B].
    """
    _require_points("centers_a", centers_a)
    _require_points("centers_b", centers_b)

    if centers_a.ndim != 2 or centers_b.ndim != 2:
        raise ValueError("circle centers must have shape [N, 2]")
    if radii_a.ndim != 1 or radii_a.shape[0] != centers_a.shape[0]:
        raise ValueError("radii_a must have shape [A]")
    if radii_b.ndim != 1 or radii_b.shape[0] != centers_b.shape[0]:
        raise ValueError("radii_b must have shape [B]")

    tensors = (centers_b, radii_a, radii_b)
    if any(t.device != centers_a.device for t in tensors):
        raise ValueError("all circle tensors must be on the same device")
    if any(t.dtype != centers_a.dtype for t in tensors):
        raise ValueError("all circle tensors must have the same dtype")
    if not radii_a.is_floating_point() or not radii_b.is_floating_point():
        raise TypeError("circle radii must have floating-point dtypes")
    if not bool(torch.isfinite(radii_a).all()) or not bool(
        torch.isfinite(radii_b).all()
    ):
        raise ValueError("circle radii must contain only finite values")
    if bool((radii_a <= 0).any()) or bool((radii_b <= 0).any()):
        raise ValueError("circle radii must be strictly positive")

    distances = torch.linalg.vector_norm(
        centers_a[:, None, :] - centers_b[None, :, :],
        dim=-1,
    )
    return distances - radii_a[:, None] - radii_b[None, :]


def circle_circle_contact(
    centers_a: Tensor,
    radii_a: Tensor,
    centers_b: Tensor,
    radii_b: Tensor,
) -> Tensor:
    """Return inclusive pairwise circle contact flags with shape [A, B]."""
    return (
        circle_circle_signed_clearance(
            centers_a, radii_a, centers_b, radii_b
        )
        <= 0
    )
