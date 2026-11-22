"""Independent per-agent objective collection bookkeeping.

Core indices refer to the stable indices in a GeneratedWorld layout.
Collection is local to each flattened agent batch entry: no core is
globally depleted, and one agent's collection never changes another's.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AgentCoreCollectionState:
    """Boolean per-agent core collection state.

    ``collected`` has shape ``batch_shape + (core_count,)``. The leading
    dimensions identify independent agents; the final dimension identifies
    cores by their stable GeneratedWorld layout index.

    The dataclass is structurally frozen. Its tensor is owned by this
    instance, but callers should treat it as read-only.
    """

    collected: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.collected, torch.Tensor):
            raise TypeError("collected must be a torch.Tensor")
        if self.collected.dtype is not torch.bool:
            raise TypeError("collected must have dtype torch.bool")
        if self.collected.ndim < 1:
            raise ValueError("collected must have a core dimension")

        # Own storage: callers retaining the input cannot mutate this state.
        object.__setattr__(self, "collected", self.collected.clone())

    @property
    def batch_shape(self) -> torch.Size:
        return self.collected.shape[:-1]

    @property
    def core_count(self) -> int:
        return self.collected.shape[-1]

    @property
    def device(self) -> torch.device:
        return self.collected.device

    @classmethod
    def initialize(
        cls,
        batch_shape: tuple[int, ...] | torch.Size,
        core_count: int,
        *,
        device: torch.device | str = "cpu",
    ) -> "AgentCoreCollectionState":
        """Create an all-uncollected state for the requested batch/layout."""
        if not isinstance(batch_shape, (tuple, torch.Size)):
            raise TypeError("batch_shape must be a tuple or torch.Size")

        normalized_shape = tuple(batch_shape)
        for dimension in normalized_shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError("batch_shape dimensions must be integers")
            if dimension < 0:
                raise ValueError("batch_shape dimensions must be nonnegative")

        if isinstance(core_count, bool) or not isinstance(core_count, int):
            raise TypeError("core_count must be an integer")
        if core_count < 0:
            raise ValueError("core_count must be nonnegative")

        try:
            target_device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise TypeError("device must identify a valid torch device") from exc

        mask = torch.zeros(
            normalized_shape + (core_count,),
            dtype=torch.bool,
            device=target_device,
        )
        return cls(mask)

    def clone(self) -> "AgentCoreCollectionState":
        """Return an independent copy of the collection state."""
        return AgentCoreCollectionState(self.collected)

    def mark_collected(
        self,
        agent_indices: torch.Tensor,
        core_indices: torch.Tensor,
    ) -> tuple["AgentCoreCollectionState", torch.Tensor]:
        """Record collection events without mutating this state.

        Event tensors contain paired, flattened agent indices and stable core
        indices. For each (agent, core) pair, only its first occurrence in
        this event batch can be newly collected.

        All index validation completes before the output mask is modified.
        Range validation uses at most two batch-level synchronizations (one
        for agent indices and one for core indices); event processing remains
        on the input device.
        """
        if not isinstance(agent_indices, torch.Tensor):
            raise TypeError("agent_indices must be a torch.Tensor")
        if not isinstance(core_indices, torch.Tensor):
            raise TypeError("core_indices must be a torch.Tensor")

        if agent_indices.dtype != torch.int64:
            raise TypeError("agent_indices must have dtype torch.int64")
        if core_indices.dtype != torch.int64:
            raise TypeError("core_indices must have dtype torch.int64")

        if agent_indices.ndim != 1 or core_indices.ndim != 1:
            raise ValueError("event index tensors must be one-dimensional")
        if agent_indices.shape != core_indices.shape:
            raise ValueError("agent_indices and core_indices must have equal shapes")

        if agent_indices.device != self.collected.device:
            raise ValueError("agent_indices must be on the collection-state device")
        if core_indices.device != self.collected.device:
            raise ValueError("core_indices must be on the collection-state device")

        event_count = agent_indices.numel()
        agent_count = math.prod(self.batch_shape)

        # Clone even for an empty event batch: the returned state must never
        # share mutable storage with the input state.
        updated_mask = self.collected.contiguous().clone()
        newly_collected = torch.zeros(
            (event_count,),
            dtype=torch.bool,
            device=self.collected.device,
        )

        if event_count == 0:
            return AgentCoreCollectionState(updated_mask), newly_collected

        # Validate both index categories before doing any updates. Each check
        # reduces the entire event batch to one scalar; there are at most two
        # data-dependent host synchronizations, never one per event.
        invalid_agents = (agent_indices < 0) | (agent_indices >= agent_count)
        invalid_cores = (core_indices < 0) | (core_indices >= self.core_count)

        # Validate each index category as a batch. At most two scalar
        # synchronizations; event arrays remain on their original device.
        if bool(invalid_agents.any().item()):
            raise IndexError("agent index out of range")
        if bool(invalid_cores.any().item()):
            raise IndexError("core index out of range")

        # A unique linear key represents each (agent, core) pair. Since the
        # indices have passed range validation, this encoding is collision-free.
        pair_keys = agent_indices * self.core_count + core_indices

        # Stable sorting keeps the original event order among equal keys.
        # The first item in each equal-key run is therefore the first
        # occurrence of that pair in the caller's event sequence.
        order = torch.argsort(pair_keys, stable=True)
        sorted_keys = pair_keys[order]

        first_in_run = torch.ones(
            (event_count,),
            dtype=torch.bool,
            device=pair_keys.device,
        )
        first_in_run[1:] = sorted_keys[1:] != sorted_keys[:-1]

        first_event_positions = order[first_in_run]
        first_pair_keys = sorted_keys[first_in_run]

        flat_updated = updated_mask.view(-1)

        # Existing pairs are not newly collected. Only first occurrences
        # whose pair was previously false are marked true in the result.
        is_new = ~flat_updated[first_pair_keys]
        newly_collected[first_event_positions] = is_new
        flat_updated[first_pair_keys] = True

        return AgentCoreCollectionState(updated_mask), newly_collected
