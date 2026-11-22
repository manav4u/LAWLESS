"""Per-agent embodiment and lifecycle state.

This module deliberately composes the frozen P0 AgentState rather than
changing it. Geometry and core ownership are not stored here.

Measured-acceleration contract
------------------------------
``measured_acceleration`` is a caller-supplied, world-frame finite-difference
measurement in m/s². For a transition from state t-1 to state t, its intended
provenance is:

    (velocity_t - velocity_t_minus_1) / dt

This module does not calculate that value and must not be seeded from P0
acceleration diagnostics. Observation-time selection and body-frame rotation
are deferred to observation assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from lawless.physics.state import AgentState


def _require_tensor(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def _has_any(mask: Tensor) -> bool:
    return bool(torch.any(mask).item())


_CONTINUOUS_PHYSICS_FIELDS = (
    "position",
    "velocity",
    "angle",
    "angular_velocity",
    "energy",
)
_SUPPORTED_PHYSICS_DTYPES = (torch.float32, torch.float64)


def _validate_agent_state(
    state: AgentState,
    *,
    name: str = "physics",
) -> None:
    """Validate the complete P0 physical-state input contract.

    This deliberately does not impose an energy/alive equivalence:
    make_state(energy=0) may validly produce alive=True until P0.step()
    applies its energy-exhaustion transition.
    """
    if not isinstance(state, AgentState):
        raise TypeError(f"{name} must be an AgentState")

    # Check tensor-ness before validate_shapes(), so malformed fields fail
    # here with a useful error rather than an incidental attribute error.
    for field in _CONTINUOUS_PHYSICS_FIELDS + ("alive",):
        _require_tensor(f"{name}.{field}", getattr(state, field))

    state.validate_shapes()

    reference = state.position
    if reference.dtype not in _SUPPORTED_PHYSICS_DTYPES:
        raise TypeError(
            f"{name}.position must use torch.float32 or torch.float64, "
            f"got {reference.dtype}"
        )

    device = reference.device
    dtype = reference.dtype

    for field in _CONTINUOUS_PHYSICS_FIELDS:
        value = getattr(state, field)

        if value.dtype not in _SUPPORTED_PHYSICS_DTYPES:
            raise TypeError(
                f"{name}.{field} must use torch.float32 or torch.float64, "
                f"got {value.dtype}"
            )
        if value.dtype != dtype:
            raise TypeError(
                f"{name}.{field} must have the same dtype as "
                f"{name}.position ({dtype}), got {value.dtype}"
            )
        if value.device != device:
            raise ValueError(
                f"{name}.{field} must be on {device}, got {value.device}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{name}.{field} must contain only finite values")

    if state.alive.dtype != torch.bool:
        raise TypeError(f"{name}.alive must have dtype torch.bool")
    if state.alive.device != device:
        raise ValueError(f"{name}.alive must be on {device}")


def _validate_mask(name: str, mask: Tensor, reference: AgentState) -> None:
    _require_tensor(name, mask)
    expected = reference.batch_shape + (1,)
    if mask.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(mask.shape)}")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if mask.device != reference.position.device:
        raise ValueError(f"{name} must be on {reference.position.device}")


def _validate_physics_compatibility(
    current: AgentState,
    replacement: AgentState,
) -> None:
    _validate_agent_state(current, name="current physics")
    _validate_agent_state(replacement, name="replacement physics")

    if replacement.batch_shape != current.batch_shape:
        raise ValueError(
            "replacement physics batch_shape must match the current state"
        )

    for name in (
        "position",
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
    ):
        old = getattr(current, name)
        new = getattr(replacement, name)
        if new.device != old.device:
            raise ValueError(f"replacement {name} must be on {old.device}")
        if new.dtype != old.dtype:
            raise ValueError(f"replacement {name} must have dtype {old.dtype}")

    if replacement.alive.device != current.alive.device:
        raise ValueError("replacement alive mask must be on the current device")


def _ensure_frozen_dead_agents(
    current: AgentState,
    replacement: AgentState,
) -> None:
    """Reject changes to physical values belonging to already-dead agents."""
    dead = ~current.alive

    for name in (
        "position",
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
    ):
        old = getattr(current, name)
        new = getattr(replacement, name)
        expanded_dead = dead.expand_as(old)
        preserved = torch.where(expanded_dead, new, old)
        if not torch.equal(preserved, old):
            raise ValueError(
                f"replacement changes frozen physical state for dead agents ({name})"
            )


@dataclass(frozen=True)
class AgentEmbodimentState:
    """Per-agent physical, action-history, measurement, and lifecycle state.

    All tensors have arbitrary leading batch dimensions matching
    ``physics.batch_shape``. The final feature dimension is 2 for vector
    quantities and 1 for scalar/mask quantities.

    ``physics.alive`` is the sole active/inactive mask. ``extracted`` and
    ``failed`` annotate terminal causes; they do not duplicate alive state.
    A dead agent with both annotations false may have exhausted its energy
    under the P0 contract.
    """

    physics: AgentState
    previous_actions: Tensor
    measured_acceleration: Tensor
    extracted: Tensor
    failed: Tensor

    def __post_init__(self) -> None:
        _validate_agent_state(self.physics)
        reference = self.physics
        batch_shape = reference.batch_shape
        vector_shape = batch_shape + (2,)
        mask_shape = batch_shape + (1,)
        floating_dtype = reference.position.dtype
        device = reference.position.device

        for name in (
            "position",
            "velocity",
            "angle",
            "angular_velocity",
            "energy",
        ):
            value = getattr(reference, name)
            if value.device != device:
                raise ValueError(f"physics.{name} must be on {device}")
            if value.dtype != floating_dtype:
                raise TypeError(
                    f"physics.{name} must have dtype {floating_dtype}"
                )

        if reference.alive.dtype != torch.bool:
            raise TypeError("physics.alive must have dtype torch.bool")
        if reference.alive.device != device:
            raise ValueError(f"physics.alive must be on {device}")

        for name in ("previous_actions", "measured_acceleration"):
            value = getattr(self, name)
            _require_tensor(name, value)
            if value.shape != vector_shape:
                raise ValueError(
                    f"{name} must have shape {vector_shape}, got {tuple(value.shape)}"
                )
            if value.dtype != floating_dtype:
                raise TypeError(f"{name} must have dtype {floating_dtype}")
            if value.device != device:
                raise ValueError(f"{name} must be on {device}")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must contain only finite values")

        if bool(
            ((self.previous_actions < -1.0) | (self.previous_actions > 1.0))
            .any()
            .item()
        ):
            raise ValueError("previous_actions must be in [-1, 1]")

        _validate_mask("extracted", self.extracted, reference)
        _validate_mask("failed", self.failed, reference)

        if _has_any(self.extracted & self.failed):
            raise ValueError("an agent cannot be both extracted and failed")

        terminal = self.extracted | self.failed
        if _has_any(terminal & reference.alive):
            raise ValueError("extracted or failed agents must not be alive")

        # Take independent ownership of all input storage. AgentState.clone()
        # is the P0-supported cloning operation.
        object.__setattr__(self, "physics", reference.clone())
        object.__setattr__(self, "previous_actions", self.previous_actions.clone())
        object.__setattr__(
            self, "measured_acceleration", self.measured_acceleration.clone()
        )
        object.__setattr__(self, "extracted", self.extracted.clone())
        object.__setattr__(self, "failed", self.failed.clone())

    @classmethod
    def initialize(cls, physics: AgentState) -> "AgentEmbodimentState":
        """Create active embodiment state from an existing physical state.

        Existing dead agents remain dead and unlabelled. In particular, this
        constructor does not reinterpret P0 energy exhaustion as failure.
        """
        _validate_agent_state(physics)
        batch_shape = physics.batch_shape
        device = physics.position.device
        dtype = physics.position.dtype

        return cls(
            physics=physics,
            previous_actions=torch.zeros(
                batch_shape + (2,), device=device, dtype=dtype
            ),
            measured_acceleration=torch.zeros(
                batch_shape + (2,), device=device, dtype=dtype
            ),
            extracted=torch.zeros(
                batch_shape + (1,), device=device, dtype=torch.bool
            ),
            failed=torch.zeros(
                batch_shape + (1,), device=device, dtype=torch.bool
            ),
        )

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.physics.batch_shape

    def clone(self) -> "AgentEmbodimentState":
        """Return a deep tensor-storage clone."""
        return AgentEmbodimentState(
            physics=self.physics.clone(),
            previous_actions=self.previous_actions,
            measured_acceleration=self.measured_acceleration,
            extracted=self.extracted,
            failed=self.failed,
        )

    def replace_physics(
        self, next_physics: AgentState
    ) -> "AgentEmbodimentState":
        """Replace physical state after a P0 transition.

        An inactive agent cannot become active again. Physical values for
        agents that were already dead must remain exactly frozen. A live
        agent may become dead through the normal P0 energy-exhaustion or
        other P0 lifecycle behavior.
        """
        _validate_physics_compatibility(self.physics, next_physics)

        resurrected = next_physics.alive & ~self.physics.alive
        if _has_any(resurrected):
            raise ValueError("physical replacement cannot reactivate dead agents")

        _ensure_frozen_dead_agents(self.physics, next_physics)

        return AgentEmbodimentState(
            physics=next_physics,
            previous_actions=self.previous_actions,
            measured_acceleration=self.measured_acceleration,
            extracted=self.extracted,
            failed=self.failed,
        )

    def record_transition_measurements(
        self,
        *,
        applied_actions: Tensor,
        measured_acceleration: Tensor,
    ) -> "AgentEmbodimentState":
        """Record explicit inputs/measurements for a completed transition.

        ``applied_actions`` must be the actual action sent to P0 after any
        clamping or preprocessing. ``measured_acceleration`` must be supplied
        by the caller from the physical state transition; P0 diagnostics are
        not read or inferred here.
        """
        # Reuse construction validation, while keeping this method's errors
        # explicit and tied to the supplied measurement contract.
        return AgentEmbodimentState(
            physics=self.physics,
            previous_actions=applied_actions,
            measured_acceleration=measured_acceleration,
            extracted=self.extracted,
            failed=self.failed,
        )

    def set_terminal(
        self,
        *,
        extracted: Optional[Tensor] = None,
        failed: Optional[Tensor] = None,
    ) -> "AgentEmbodimentState":
        """Apply terminal annotations to currently active agents.

        Repeating the same annotation is idempotent. Conflicting terminal
        causes, or annotating an energy-exhausted/already-terminal agent with
        a different cause, are rejected. The physical alive mask is set false
        for newly terminal agents.
        """
        reference = self.physics
        shape = reference.batch_shape + (1,)
        device = reference.position.device

        if extracted is None:
            extracted = torch.zeros(shape, dtype=torch.bool, device=device)
        if failed is None:
            failed = torch.zeros(shape, dtype=torch.bool, device=device)

        _validate_mask("extracted request", extracted, reference)
        _validate_mask("failed request", failed, reference)

        if _has_any(extracted & failed):
            raise ValueError("an agent cannot be extracted and failed together")

        # A repeated same-cause request is harmless. A request for the
        # opposite cause, or for an unlabelled inactive agent, is invalid.
        invalid_extracted = extracted & (
            self.failed | (~reference.alive & ~self.extracted)
        )
        invalid_failed = failed & (
            self.extracted | (~reference.alive & ~self.failed)
        )
        if _has_any(invalid_extracted):
            raise ValueError("cannot extract an already failed or inactive agent")
        if _has_any(invalid_failed):
            raise ValueError("cannot fail an already extracted or inactive agent")

        next_extracted = self.extracted | extracted
        next_failed = self.failed | failed
        newly_terminal = extracted | failed

        next_physics = AgentState(
            position=reference.position,
            velocity=reference.velocity,
            angle=reference.angle,
            angular_velocity=reference.angular_velocity,
            energy=reference.energy,
            alive=reference.alive & ~newly_terminal,
        )

        return AgentEmbodimentState(
            physics=next_physics,
            previous_actions=self.previous_actions,
            measured_acceleration=self.measured_acceleration,
            extracted=next_extracted,
            failed=next_failed,
        )

    def mark_extracted(self, mask: Tensor) -> "AgentEmbodimentState":
        return self.set_terminal(extracted=mask)

    def mark_failed(self, mask: Tensor) -> "AgentEmbodimentState":
        return self.set_terminal(failed=mask)
