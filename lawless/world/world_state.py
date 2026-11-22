"""Single-world state for the LAWLESS P1 environment.

A WorldState contains rectangular world dimensions and one shared
static circle-obstacle layout. It does not generate obstacles or
perform collision response.

Coordinates and dimensions are in metres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from lawless.world.boundaries import boundary_clearance, inside_world
from lawless.world.config import WorldConfig
from lawless.world.geometry import (
    CircleObstacles,
    point_circle_signed_clearance,
)


@dataclass(frozen=True)
class WorldState:
    """Geometry for one world.

    width_m, height_m:
        Positive finite world dimensions.

    obstacles:
        Shared static obstacle layout:
        centers [M, 2], radii [M].

    Obstacle circles must be fully contained in the closed world
    rectangle. Zero obstacles are valid.

    The dataclass prevents field reassignment, but does not make the
    underlying PyTorch tensor storage immutable.
    """

    width_m: float
    height_m: float
    obstacles: CircleObstacles

    def __post_init__(self) -> None:
        for name, value in (
            ("width_m", self.width_m),
            ("height_m", self.height_m),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and strictly positive"
                )

        if not isinstance(self.obstacles, CircleObstacles):
            raise TypeError("obstacles must be a CircleObstacles instance")

        centers = self.obstacles.centers
        radii = self.obstacles.radii

        # A circle is contained only if its entire radius remains
        # within all four closed world boundaries.
        contained = (
            (centers[:, 0] - radii >= 0)
            & (centers[:, 0] + radii <= self.width_m)
            & (centers[:, 1] - radii >= 0)
            & (centers[:, 1] + radii <= self.height_m)
        )
        if not bool(contained.all()):
            raise ValueError(
                "every obstacle circle must be fully contained "
                "within the world boundaries"
            )

    @classmethod
    def from_config(
        cls,
        config: WorldConfig,
        obstacles: CircleObstacles,
    ) -> "WorldState":
        """Construct a world from WorldConfig and an explicit layout.

        The obstacle count must match config.obstacle_count. Placement
        and random generation are deliberately outside this method.
        """
        if not isinstance(config, WorldConfig):
            raise TypeError("config must be a WorldConfig instance")

        config.validate()

        if not isinstance(obstacles, CircleObstacles):
            raise TypeError("obstacles must be a CircleObstacles instance")

        if obstacles.count != config.obstacle_count:
            raise ValueError(
                "obstacle count does not match "
                "config.obstacle_count"
            )

        return cls(
            width_m=config.width_m,
            height_m=config.height_m,
            obstacles=obstacles,
        )

    @property
    def obstacle_count(self) -> int:
        """Number of static circles in this world."""
        return self.obstacles.count

    def contains_positions(self, positions: Tensor) -> Tensor:
        """Return boundary-inclusive membership for positions [..., 2].

        Output shape is [...], boolean dtype, on positions.device.
        """
        return inside_world(
            positions,
            width_m=self.width_m,
            height_m=self.height_m,
        )

    def boundary_clearance(self, positions: Tensor) -> Tensor:
        """Return minimum signed axis clearance for positions [..., 2].

        This is not Euclidean distance outside rectangle corners.
        Output shape is [...], preserving positions dtype and device.
        """
        return boundary_clearance(
            positions,
            width_m=self.width_m,
            height_m=self.height_m,
        )

    def obstacle_clearance(self, positions: Tensor) -> Tensor:
        """Return point-to-obstacle signed clearance [..., M].

        Positive means separated, zero means tangent, and negative
        means the point lies inside the obstacle circle.

        With zero obstacles, output shape is [..., 0].
        """
        return point_circle_signed_clearance(
            positions,
            self.obstacles,
        )
