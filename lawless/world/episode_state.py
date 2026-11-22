"""Per-agent episode state composed over a shared generated layout.

This module owns per-agent physical, embodiment, spawn-assignment, and
objective bookkeeping. GeneratedWorld remains shared and read-only by
contract; this module does not modify its geometry.

An initialized measured_acceleration buffer contains zeros because no
completed transition measurement exists yet. Those zeros are an
uninitialized-history sentinel by convention, not a measured zero
acceleration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lawless.physics.constants import PhysicsConfig
from lawless.physics.state import AgentState, make_state
from lawless.world.config import EmbodimentConfig, ObjectiveConfig
from lawless.world.embodiment import AgentEmbodimentState
from lawless.world.generated_world import GeneratedWorld
from lawless.world.objective_state import AgentCoreCollectionState


def _validate_layout_and_config(
    generated_world: GeneratedWorld,
    embodiment_config: EmbodimentConfig,
    objective_config: ObjectiveConfig,
) -> tuple[torch.device, torch.dtype, int]:
    if not isinstance(generated_world, GeneratedWorld):
        raise TypeError("generated_world must be a GeneratedWorld")
    if not isinstance(embodiment_config, EmbodimentConfig):
        raise TypeError("embodiment_config must be an EmbodimentConfig")
    if not isinstance(objective_config, ObjectiveConfig):
        raise TypeError("objective_config must be an ObjectiveConfig")

    # Use the canonical section validators before any lossy conversion.
    embodiment_config.validate()
    objective_config.validate()

    positions = generated_world.spawn_positions
    cores = generated_world.core_positions
    if not isinstance(positions, Tensor) or not isinstance(cores, Tensor):
        raise TypeError("generated layout positions must be tensors")
    if positions.ndim != 2 or positions.shape[-1] != 2:
        raise ValueError("generated_world.spawn_positions must have shape [S, 2]")
    if cores.ndim != 2 or cores.shape[-1] != 2:
        raise ValueError("generated_world.core_positions must have shape [K, 2]")

    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("generated layout positions must use float32 or float64")
    if cores.dtype != positions.dtype:
        raise TypeError("core_positions must have the spawn-position dtype")
    if cores.device != positions.device:
        raise ValueError("core_positions and spawn_positions must share a device")

    core_count = int(cores.shape[0])
    if objective_config.core_count != core_count:
        raise ValueError(
            "objective_config.core_count must match generated_world.core_positions"
        )

    # The config validator has already rejected strings, bools, non-finite
    # values, and non-positive radii. Compare the radius after representation
    # in the layout dtype, as that is the physical value used by the layout.
    configured_radius = torch.tensor(
        embodiment_config.agent_radius_m,
        dtype=positions.dtype,
        device=positions.device,
    )
    represented_radius = float(configured_radius.item())
    if represented_radius != float(generated_world.agent_body_radius_m):
        raise ValueError(
            "embodiment_config.agent_radius_m does not match the "
            "represented generated-world agent_body_radius_m"
        )

    return positions.device, positions.dtype, core_count


def _validate_spawn_indices(
    spawn_indices: Tensor,
    *,
    generated_world: GeneratedWorld,
    batch_shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(spawn_indices, Tensor):
        raise TypeError("spawn_indices must be a torch.Tensor")
    if spawn_indices.dtype != torch.int64:
        raise TypeError("spawn_indices must have dtype torch.int64")
    if tuple(spawn_indices.shape) != batch_shape:
        raise ValueError(
            f"spawn_indices must have exact batch shape {batch_shape}, "
            f"got {tuple(spawn_indices.shape)}"
        )
    if spawn_indices.device != device:
        raise ValueError(f"spawn_indices must be on {device}")

    if spawn_indices.numel() == 0:
        raise ValueError("spawn_indices must assign at least one agent")

    invalid = (spawn_indices < 0) | (
        spawn_indices >= generated_world.spawn_positions.shape[0]
    )
    if bool(invalid.any().item()):
        raise IndexError("spawn index out of range")


@dataclass(frozen=True)
class AgentEpisodeState:
    """Independent per-agent episode state over one shared generated layout.

    ``generated_world`` is deliberately retained by reference. All other
    tensor-bearing state is independently owned by this instance.

    ``spawn_indices`` records the caller's explicit spawn-slot assignment
    with the same leading dimensions as the agent batch.
    """

    generated_world: GeneratedWorld
    spawn_indices: Tensor
    embodiment: AgentEmbodimentState
    objectives: AgentCoreCollectionState

    def __post_init__(self) -> None:
        if not isinstance(self.generated_world, GeneratedWorld):
            raise TypeError("generated_world must be a GeneratedWorld")
        if not isinstance(self.embodiment, AgentEmbodimentState):
            raise TypeError("embodiment must be an AgentEmbodimentState")
        if not isinstance(self.objectives, AgentCoreCollectionState):
            raise TypeError("objectives must be an AgentCoreCollectionState")
        if not isinstance(self.spawn_indices, Tensor):
            raise TypeError("spawn_indices must be a torch.Tensor")

        batch_shape = self.embodiment.batch_shape
        if tuple(self.spawn_indices.shape) != batch_shape:
            raise ValueError("spawn_indices shape must match embodiment batch_shape")
        if self.spawn_indices.dtype != torch.int64:
            raise TypeError("spawn_indices must have dtype torch.int64")

        physics = self.embodiment.physics
        layout_reference = self.generated_world.spawn_positions
        if physics.position.dtype != layout_reference.dtype:
            raise TypeError(
                "physics dtype must match generated-world layout dtype "
                f"({layout_reference.dtype}), got {physics.position.dtype}"
            )
        if physics.position.device != layout_reference.device:
            raise ValueError(
                "physics device must match generated-world layout device "
                f"({layout_reference.device}), got {physics.position.device}"
            )
        if self.spawn_indices.device != physics.position.device:
            raise ValueError("spawn_indices must be on the physics device")

        if self.spawn_indices.ndim < 1 or any(
            dimension <= 0 for dimension in self.spawn_indices.shape
        ):
            raise ValueError("spawn_indices must have positive batch dimensions")

        spawn_count = self.generated_world.spawn_positions.shape[0]
        invalid_spawn_indices = (self.spawn_indices < 0) | (
            self.spawn_indices >= spawn_count
        )
        if bool(invalid_spawn_indices.any().item()):
            raise IndexError("spawn index out of range")
        if self.objectives.batch_shape != batch_shape:
            raise ValueError("objective batch_shape must match embodiment batch_shape")
        if self.objectives.device != physics.position.device:
            raise ValueError("objective state must be on the physics device")
        if self.objectives.core_count != self.generated_world.core_positions.shape[0]:
            raise ValueError("objective core_count must match the generated layout")

        # This composition owns its per-agent tensors. The shared layout is
        # intentionally not cloned or reconstructed.
        object.__setattr__(self, "spawn_indices", self.spawn_indices.clone())
        object.__setattr__(self, "embodiment", self.embodiment.clone())
        object.__setattr__(self, "objectives", self.objectives.clone())

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.embodiment.batch_shape

    @property
    def physics(self) -> AgentState:
        """The P0 physical state composed by the embodiment."""
        return self.embodiment.physics

    @classmethod
    def initialize(
        cls,
        generated_world: GeneratedWorld,
        spawn_indices: Tensor,
        *,
        embodiment_config: EmbodimentConfig,
        objective_config: ObjectiveConfig,
        physics_config: PhysicsConfig,
    ) -> "AgentEpisodeState":
        """Initialize independent agent state at explicit layout spawn slots.

        ``spawn_indices`` must have exactly the intended agent batch shape.
        No implicit broadcasting, spawn selection, or assignment generation
        occurs here.
        """
        if not isinstance(physics_config, PhysicsConfig):
            raise TypeError("physics_config must be a PhysicsConfig")

        device, dtype, core_count = _validate_layout_and_config(
            generated_world,
            embodiment_config,
            objective_config,
        )

        if not isinstance(spawn_indices, Tensor):
            raise TypeError("spawn_indices must be a torch.Tensor")
        if spawn_indices.ndim < 1:
            raise ValueError("spawn_indices must have at least one batch dimension")
        batch_shape = tuple(spawn_indices.shape)
        if any(dimension <= 0 for dimension in batch_shape):
            raise ValueError("spawn_indices batch dimensions must be positive")

        _validate_spawn_indices(
            spawn_indices,
            generated_world=generated_world,
            batch_shape=batch_shape,
            device=device,
        )

        physics = make_state(
            batch_shape,
            device=device,
            dtype=dtype,
            energy=physics_config.initial_energy,
        )

        # Gather the explicitly assigned slots, preserving the requested
        # batch shape. Clone the result so physical position storage is
        # independent of the shared GeneratedWorld spawn-position tensor.
        selected_positions = generated_world.spawn_positions.index_select(
            0, spawn_indices.reshape(-1)
        ).reshape(batch_shape + (2,))

        physics = AgentState(
            position=selected_positions.clone(),
            velocity=physics.velocity,
            angle=physics.angle,
            angular_velocity=physics.angular_velocity,
            energy=physics.energy,
            alive=physics.alive,
        )

        embodiment = AgentEmbodimentState.initialize(physics)
        objectives = AgentCoreCollectionState.initialize(
            batch_shape,
            core_count,
            device=device,
        )

        return cls(
            generated_world=generated_world,
            spawn_indices=spawn_indices,
            embodiment=embodiment,
            objectives=objectives,
        )

    @classmethod
    def reset(
        cls,
        generated_world: GeneratedWorld,
        spawn_indices: Tensor,
        *,
        embodiment_config: EmbodimentConfig,
        objective_config: ObjectiveConfig,
        physics_config: PhysicsConfig,
    ) -> "AgentEpisodeState":
        """Create a fresh episode from explicit spawn assignments.

        Reset is construction of a new state, not mutation or implicit reuse
        of the previous episode's physical, history, objective, or lifecycle
        tensors. The caller must supply the spawn assignment again.
        """
        return cls.initialize(
            generated_world,
            spawn_indices,
            embodiment_config=embodiment_config,
            objective_config=objective_config,
            physics_config=physics_config,
        )

    def clone(self) -> "AgentEpisodeState":
        """Return independent per-agent storage while sharing the layout."""
        return AgentEpisodeState(
            generated_world=self.generated_world,
            spawn_indices=self.spawn_indices,
            embodiment=self.embodiment,
            objectives=self.objectives,
        )

    def replace_physics(self, next_physics: AgentState) -> "AgentEpisodeState":
        """Replace physics while preserving history, objectives, and lifecycle.

        Dead-state freezing and no-resurrection validation are delegated to
        AgentEmbodimentState.replace_physics.
        """
        next_embodiment = self.embodiment.replace_physics(next_physics)
        return AgentEpisodeState(
            generated_world=self.generated_world,
            spawn_indices=self.spawn_indices,
            embodiment=next_embodiment,
            objectives=self.objectives,
        )

    def record_transition_measurements(
        self,
        *,
        applied_actions: Tensor,
        measured_acceleration: Tensor,
    ) -> "AgentEpisodeState":
        """Record caller-supplied action and finite-difference measurement."""
        next_embodiment = self.embodiment.record_transition_measurements(
            applied_actions=applied_actions,
            measured_acceleration=measured_acceleration,
        )
        return AgentEpisodeState(
            generated_world=self.generated_world,
            spawn_indices=self.spawn_indices,
            embodiment=next_embodiment,
            objectives=self.objectives,
        )

    def mark_collected(
        self,
        agent_indices: Tensor,
        core_indices: Tensor,
    ) -> tuple["AgentEpisodeState", Tensor]:
        """Record already-decided collection events, without eligibility logic."""
        next_objectives, newly_collected = self.objectives.mark_collected(
            agent_indices,
            core_indices,
        )
        next_state = AgentEpisodeState(
            generated_world=self.generated_world,
            spawn_indices=self.spawn_indices,
            embodiment=self.embodiment,
            objectives=next_objectives,
        )
        return next_state, newly_collected

    def set_terminal(
        self,
        *,
        extracted: Tensor | None = None,
        failed: Tensor | None = None,
    ) -> "AgentEpisodeState":
        """Apply explicit terminal annotations using embodiment lifecycle rules."""
        next_embodiment = self.embodiment.set_terminal(
            extracted=extracted,
            failed=failed,
        )
        return AgentEpisodeState(
            generated_world=self.generated_world,
            spawn_indices=self.spawn_indices,
            embodiment=next_embodiment,
            objectives=self.objectives,
        )

    def mark_extracted(self, mask: Tensor) -> "AgentEpisodeState":
        return self.set_terminal(extracted=mask)

    def mark_failed(self, mask: Tensor) -> "AgentEpisodeState":
        return self.set_terminal(failed=mask)
