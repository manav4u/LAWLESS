"""Immutable composition of independent LAWLESS episode worlds.

The leading environment dimension indexes independent GeneratedWorld layouts.
Each world contains N paired candidate rollouts over the same static geometry
and the same explicitly assigned spawn slot. Agent physical and objective
state remains independent within each world. Candidates are independent
rollouts, not simultaneous competitors: there is no agent-agent collision,
and core collection is tracked independently for each candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Tuple
import math

import torch
from torch import Tensor

from lawless.physics.constants import PhysicsConfig
from lawless.world.config import EnvironmentConfig
from lawless.world.episode_state import AgentEpisodeState


@dataclass(frozen=True)
class WorldEpisodeState:
    """One world's episode clock, conditions, and N independent rollouts."""

    episode: AgentEpisodeState
    physics_config: PhysicsConfig
    gravity: Tensor
    shift_direction: int
    step_count: int = 0
    shift_applied: bool = False
    truncated: Tensor | None = None
    completed: bool = False
    world_seed: int = 0
    condition_seed: int = 0
    initial_gravity: Tensor | None = None
    linear_damping_s_inv: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.episode, AgentEpisodeState):
            raise TypeError("episode must be an AgentEpisodeState")
        if len(self.episode.batch_shape) != 1:
            raise ValueError("a world episode must have exactly one candidate dimension")
        if not isinstance(self.physics_config, PhysicsConfig):
            raise TypeError("physics_config must be a PhysicsConfig")
        if not isinstance(self.gravity, Tensor) or self.gravity.shape != (2,):
            raise ValueError("gravity must be a tensor with shape (2,)")
        if self.gravity.dtype != self.episode.physics.position.dtype:
            raise TypeError("gravity dtype must match episode physics")
        if self.gravity.device != self.episode.physics.position.device:
            raise ValueError("gravity device must match episode physics")
        if not bool(torch.isfinite(self.gravity).all().item()):
            raise ValueError("gravity must be finite")
        if (
            isinstance(self.shift_direction, bool)
            or not isinstance(self.shift_direction, Integral)
            or int(self.shift_direction) not in (-1, 1)
        ):
            raise ValueError("shift_direction must be -1 or +1")
        for name in ("world_seed", "condition_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= int(value) < (1 << 63):
                raise ValueError(f"{name} must be a nonnegative 63-bit integer")
        if (
            isinstance(self.linear_damping_s_inv, bool)
            or not isinstance(self.linear_damping_s_inv, (int, float))
            or not math.isfinite(float(self.linear_damping_s_inv))
            or self.linear_damping_s_inv < 0
        ):
            raise ValueError("linear_damping_s_inv must be finite and nonnegative")
        if isinstance(self.step_count, bool) or not isinstance(self.step_count, int) or self.step_count < 0:
            raise ValueError("step_count must be a nonnegative integer")
        if not isinstance(self.shift_applied, bool) or not isinstance(self.completed, bool):
            raise TypeError("shift_applied and completed must be bool")
        expected = tuple(self.episode.batch_shape) + (1,)
        truncated = self.truncated
        if truncated is None:
            truncated = torch.zeros(expected, dtype=torch.bool, device=self.gravity.device)
        if not isinstance(truncated, Tensor) or tuple(truncated.shape) != expected:
            raise ValueError(f"truncated must have shape {expected}")
        if truncated.dtype != torch.bool or truncated.device != self.gravity.device:
            raise TypeError("truncated must be bool on the physics device")
        if bool((truncated & self.episode.physics.alive.logical_not()).any().item()):
            raise ValueError("physically inactive agents cannot be truncated")
        if bool(truncated.any().item()) and not self.completed:
            raise ValueError("a world with truncated agents must be completed")
        if not self.completed and not bool(self.episode.physics.alive.any().item()):
            raise ValueError("a world with no physically active agents must be completed")
        if self.completed and not bool(
            ((~self.episode.physics.alive) | truncated).all().item()
        ):
            raise ValueError("completed worlds must have every agent terminated or truncated")
        initial = self.gravity if self.initial_gravity is None else self.initial_gravity
        if not isinstance(initial, Tensor) or initial.shape != (2,):
            raise ValueError("initial_gravity must have shape (2,)")
        if initial.dtype != self.gravity.dtype or initial.device != self.gravity.device:
            raise ValueError("initial_gravity must match gravity dtype and device")
        if not bool(torch.isfinite(initial).all().item()):
            raise ValueError("initial_gravity must be finite")
        if self.episode.generated_world.seed != int(self.world_seed):
            raise ValueError("world_seed must match the GeneratedWorld seed")
        if not math.isclose(
            self.physics_config.linear_damping,
            float(self.linear_damping_s_inv),
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("physics_config linear damping must match sampled episode damping")
        object.__setattr__(self, "gravity", self.gravity.clone())
        object.__setattr__(self, "initial_gravity", initial.clone())
        object.__setattr__(self, "truncated", truncated.clone())

    @property
    def num_agents(self) -> int:
        return int(self.episode.batch_shape[0])

    @property
    def terminated(self) -> Tensor:
        """Physical inactivity, including unlabelled P0 energy exhaustion."""
        return ~self.episode.physics.alive.squeeze(-1)


@dataclass(frozen=True)
class EnvironmentState:
    """State for B independent worlds, each with N candidate rollouts."""

    worlds: Tuple[WorldEpisodeState, ...]
    config: EnvironmentConfig
    num_agents: int

    def __post_init__(self) -> None:
        if not isinstance(self.worlds, tuple) or len(self.worlds) == 0:
            raise ValueError("worlds must be a nonempty tuple")
        if isinstance(self.num_agents, bool) or not isinstance(self.num_agents, int) or self.num_agents <= 0:
            raise ValueError("num_agents must be a positive integer")
        if not isinstance(self.config, EnvironmentConfig):
            raise TypeError("config must be an EnvironmentConfig")
        self.config.validate()
        if any(not isinstance(world, WorldEpisodeState) for world in self.worlds):
            raise TypeError("every world must be a WorldEpisodeState")
        reference = self.worlds[0].episode.physics.position
        reference_core_count = self.worlds[0].episode.generated_world.core_positions.shape[0]
        for world in self.worlds:
            if world.num_agents != self.num_agents:
                raise ValueError("all worlds must have the same candidate count")
            if not math.isclose(
                world.physics_config.dt,
                self.config.episode.dt_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("world timestep must match EnvironmentConfig")
            if not math.isclose(
                world.physics_config.angular_damping,
                self.config.episode.angular_damping_s_inv,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("world angular damping must match EpisodeConfig")
            if world.episode.physics.position.device != reference.device:
                raise ValueError("all worlds must use the same physics device")
            if world.episode.physics.position.dtype != reference.dtype:
                raise ValueError("all worlds must use the same physics dtype")
            if world.episode.generated_world.core_positions.shape[0] != reference_core_count:
                raise ValueError("all worlds must use the same core count")
        object.__setattr__(self, "worlds", tuple(self.worlds))

    @property
    def num_worlds(self) -> int:
        return len(self.worlds)

    @property
    def batch_shape(self) -> tuple[int, int]:
        return (self.num_worlds, self.num_agents)

    @property
    def completed(self) -> Tensor:
        return torch.tensor(
            [world.completed for world in self.worlds],
            dtype=torch.bool,
            device=self.worlds[0].gravity.device,
        )

    @property
    def all_completed(self) -> bool:
        return all(world.completed for world in self.worlds)
