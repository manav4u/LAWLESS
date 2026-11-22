"""Structurally frozen record for one complete generated world layout.

Tensor storage remains mutable in PyTorch. Callers must treat the layout
tensors as read-only; mutable episode state belongs in later environment
integration stages.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from lawless.world.world_state import WorldState


@dataclass(frozen=True)
class GeneratedWorld:
    """One world geometry and its complete initial-layout data.

    Scalar radii are stored as Python floats whose values are first rounded
    to the selected tensor dtype. Thus a configured decimal such as 0.1
    means the nearest representable float32 or float64 value, not exact
    real-number equality with that decimal.

    Tensor shapes:
        spawn_positions: [S, 2]
        extraction_center: [2]
        core_positions: [K, 2]
        core_radii: [K]

    All layout tensors must share the same floating dtype and device as the
    obstacle tensors in ``world``.
    """

    world: WorldState
    spawn_positions: torch.Tensor
    agent_body_radius_m: float
    extraction_center: torch.Tensor
    extraction_radius_m: float
    core_positions: torch.Tensor
    core_radii: torch.Tensor
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.world, WorldState):
            raise TypeError("world must be a WorldState")

        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed <= (1 << 63) - 1:
            raise ValueError("seed must be in [0, 2**63 - 1]")

        obstacles = self.world.obstacles
        reference_centers = obstacles.centers
        reference_radii = obstacles.radii

        if reference_centers.dtype not in (torch.float32, torch.float64):
            raise TypeError("layout dtype must be torch.float32 or torch.float64")
        if reference_centers.device != reference_radii.device:
            raise ValueError("world obstacle tensors must share a device")
        if reference_centers.dtype != reference_radii.dtype:
            raise ValueError("world obstacle tensors must share a dtype")

        dtype = reference_centers.dtype
        device = reference_centers.device

        tensor_fields = {
            "spawn_positions": self.spawn_positions,
            "extraction_center": self.extraction_center,
            "core_positions": self.core_positions,
            "core_radii": self.core_radii,
        }
        for name, tensor in tensor_fields.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.dtype != dtype:
                raise ValueError(f"{name} must use the world obstacle dtype")
            if tensor.device != device:
                raise ValueError(f"{name} must use the world obstacle device")
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} must contain only finite values")

        if self.spawn_positions.ndim != 2 or self.spawn_positions.shape[1] != 2:
            raise ValueError("spawn_positions must have shape [S, 2]")
        if self.extraction_center.shape != (2,):
            raise ValueError("extraction_center must have shape [2]")
        if self.core_positions.ndim != 2 or self.core_positions.shape[1] != 2:
            raise ValueError("core_positions must have shape [K, 2]")
        if self.core_radii.ndim != 1:
            raise ValueError("core_radii must have shape [K]")
        if self.core_radii.shape[0] != self.core_positions.shape[0]:
            raise ValueError("core_radii length must match core_positions")

        agent_radius = self._validate_scalar_radius(
            "agent_body_radius_m", self.agent_body_radius_m, dtype
        )
        extraction_radius = self._validate_scalar_radius(
            "extraction_radius_m", self.extraction_radius_m, dtype
        )

        # Frozen dataclasses do not make tensor storage immutable. Normalize
        # scalar values to the exact numeric values represented by the dtype.
        object.__setattr__(self, "agent_body_radius_m", agent_radius)
        object.__setattr__(self, "extraction_radius_m", extraction_radius)

        if not bool(torch.isfinite(reference_centers).all()):
            raise ValueError("obstacle centers must be finite")
        if not bool(torch.isfinite(reference_radii).all()):
            raise ValueError("obstacle radii must be finite")
        if not bool((reference_radii > 0).all()):
            raise ValueError("obstacle radii must be positive")

        if not bool((self.core_radii > 0).all()):
            raise ValueError("core radii must be positive")

        width = float(self.world.width_m)
        height = float(self.world.height_m)

        if not math.isfinite(width) or not math.isfinite(height):
            raise ValueError("world dimensions must be finite")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("world dimensions must be positive")

        self._validate_circle_containment(
            reference_centers,
            reference_radii,
            width,
            height,
            "obstacle",
        )
        self._validate_circle_containment(
            self.spawn_positions,
            torch.full(
                (self.spawn_positions.shape[0],),
                agent_radius,
                dtype=dtype,
                device=device,
            ),
            width,
            height,
            "spawn body",
        )
        self._validate_circle_containment(
            self.extraction_center.reshape(1, 2),
            torch.tensor([extraction_radius], dtype=dtype, device=device),
            width,
            height,
            "extraction",
        )
        self._validate_circle_containment(
            self.core_positions,
            self.core_radii,
            width,
            height,
            "core",
        )

        self._validate_pairwise_clearance(
            reference_centers,
            reference_radii,
            reference_centers,
            reference_radii,
            same_set=True,
            label="obstacle-obstacle",
        )
        self._validate_pairwise_clearance(
            self.spawn_positions,
            torch.full(
                (self.spawn_positions.shape[0],),
                agent_radius,
                dtype=dtype,
                device=device,
            ),
            reference_centers,
            reference_radii,
            label="spawn-obstacle",
        )
        self._validate_pairwise_clearance(
            self.spawn_positions,
            torch.full(
                (self.spawn_positions.shape[0],),
                agent_radius,
                dtype=dtype,
                device=device,
            ),
            self.spawn_positions,
            torch.full(
                (self.spawn_positions.shape[0],),
                agent_radius,
                dtype=dtype,
                device=device,
            ),
            same_set=True,
            label="spawn-spawn",
        )
        self._validate_pairwise_clearance(
            self.extraction_center.reshape(1, 2),
            torch.tensor([extraction_radius], dtype=dtype, device=device),
            reference_centers,
            reference_radii,
            label="extraction-obstacle",
        )
        self._validate_pairwise_clearance(
            self.spawn_positions,
            torch.full(
                (self.spawn_positions.shape[0],),
                agent_radius,
                dtype=dtype,
                device=device,
            ),
            self.extraction_center.reshape(1, 2),
            torch.tensor([extraction_radius], dtype=dtype, device=device),
            label="spawn-extraction",
        )
        self._validate_pairwise_clearance(
            self.core_positions,
            self.core_radii,
            reference_centers,
            reference_radii,
            label="core-obstacle",
        )
        self._validate_pairwise_clearance(
            self.core_positions,
            self.core_radii,
            self.spawn_positions,
            torch.full(
                (self.spawn_positions.shape[0],),
                agent_radius,
                dtype=dtype,
                device=device,
            ),
            label="core-spawn",
        )
        self._validate_pairwise_clearance(
            self.core_positions,
            self.core_radii,
            self.extraction_center.reshape(1, 2),
            torch.tensor([extraction_radius], dtype=dtype, device=device),
            label="core-extraction",
        )
        self._validate_pairwise_clearance(
            self.core_positions,
            self.core_radii,
            self.core_positions,
            self.core_radii,
            same_set=True,
            label="core-core",
        )

    @staticmethod
    def _validate_scalar_radius(
        name: str,
        value: float,
        dtype: torch.dtype,
    ) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite positive real number")

        numeric_value = float(value)
        if not math.isfinite(numeric_value) or numeric_value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")

        rounded = torch.tensor(numeric_value, dtype=dtype, device="cpu")
        if not bool(torch.isfinite(rounded)) or not bool(rounded > 0):
            raise ValueError(
                f"{name} must remain finite and positive in the layout dtype"
            )
        return float(rounded.item())

    @staticmethod
    def _validate_circle_containment(
        centers: torch.Tensor,
        radii: torch.Tensor,
        width: float,
        height: float,
        label: str,
    ) -> None:
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
            raise ValueError(f"{label} circle is not fully inside the world")

    @staticmethod
    def _validate_pairwise_clearance(
        centers_a: torch.Tensor,
        radii_a: torch.Tensor,
        centers_b: torch.Tensor,
        radii_b: torch.Tensor,
        *,
        label: str,
        same_set: bool = False,
    ) -> None:
        if centers_a.shape[0] == 0 or centers_b.shape[0] == 0:
            return

        delta = centers_a[:, None, :] - centers_b[None, :, :]
        distances = torch.linalg.vector_norm(delta, dim=-1)
        required = radii_a[:, None] + radii_b[None, :]

        if same_set:
            relevant = torch.triu(
                torch.ones_like(distances, dtype=torch.bool),
                diagonal=1,
            )
            if not bool((distances[relevant] > required[relevant]).all()):
                raise ValueError(f"{label} circles overlap or touch")
        elif not bool((distances > required).all()):
            raise ValueError(f"{label} circles overlap or touch")
