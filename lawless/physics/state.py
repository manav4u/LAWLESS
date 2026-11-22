"""Batched physical state containers."""
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class AgentState:
    position: torch.Tensor       # [..., 2], metres
    velocity: torch.Tensor       # [..., 2], metres / second
    angle: torch.Tensor          # [..., 1], radians
    angular_velocity: torch.Tensor  # [..., 1], radians / second
    energy: torch.Tensor         # [..., 1], joules
    alive: torch.Tensor          # [..., 1], bool

    @property
    def batch_shape(self) -> torch.Size:
        return self.position.shape[:-1]

    def validate_shapes(self) -> None:
        prefix = self.position.shape[:-1]
        expected = {
            "position": (2,), "velocity": (2,), "angle": (1,),
            "angular_velocity": (1,), "energy": (1,), "alive": (1,),
        }
        for name, tail in expected.items():
            value = getattr(self, name)
            if value.shape != prefix + tail:
                raise ValueError(f"{name} shape {tuple(value.shape)} != {prefix + tail}")
        if self.alive.dtype != torch.bool:
            raise TypeError("alive must have torch.bool dtype")

    def clone(self) -> "AgentState":
        return AgentState(**{name: getattr(self, name).clone() for name in self.__dataclass_fields__})

    def to(self, *args: Any, **kwargs: Any) -> "AgentState":
        # Keep alive mask boolean even when floating dtype is changed.
        values = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            values[name] = value.to(*args, **kwargs) if name != "alive" else value.to(device=kwargs.get("device", args[0] if args and isinstance(args[0], (str, torch.device)) else value.device))
        return AgentState(**values)


def make_state(
    leading_shape: tuple[int, ...] | int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    energy: float = 100.0,
) -> AgentState:
    """Create zeroed agents; `leading_shape` may be N or (B, N)."""
    if isinstance(leading_shape, int):
        leading_shape = (leading_shape,)
    if not leading_shape or any(int(x) <= 0 for x in leading_shape):
        raise ValueError("leading_shape must contain positive dimensions")
    shape = tuple(int(x) for x in leading_shape)
    state = AgentState(
        position=torch.zeros((*shape, 2), device=device, dtype=dtype),
        velocity=torch.zeros((*shape, 2), device=device, dtype=dtype),
        angle=torch.zeros((*shape, 1), device=device, dtype=dtype),
        angular_velocity=torch.zeros((*shape, 1), device=device, dtype=dtype),
        energy=torch.full((*shape, 1), float(energy), device=device, dtype=dtype),
        alive=torch.ones((*shape, 1), device=device, dtype=torch.bool),
    )
    state.validate_shapes()
    return state
