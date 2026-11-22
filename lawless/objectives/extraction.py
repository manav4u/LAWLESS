"""Extraction detection and successful episode termination."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from lawless.objectives._validation import (
    _validate_episode_and_resolution,
)
from lawless.world.collision import CollisionResolution
from lawless.world.config import EpisodeConfig
from lawless.world.episode_state import AgentEpisodeState
from lawless.world.generated_world import GeneratedWorld


@dataclass(frozen=True)
class ExtractionDetectionResult:
    """Read-only per-agent extraction eligibility and success decisions."""

    eligible: Tensor
    within_radius: Tensor
    within_time: Tensor
    success: Tensor
    generated_world: GeneratedWorld
    elapsed_time_s: float


@dataclass(frozen=True)
class ExtractionApplicationResult:
    """Episode state and freshly recomputed extraction decision."""

    episode_state: AgentEpisodeState
    detection: ExtractionDetectionResult
    newly_extracted: Tensor


def _validate_elapsed_time(elapsed_time_s: float | int) -> float:
    """Validate explicit time without converting it to physics tensor dtype."""

    if isinstance(elapsed_time_s, bool) or not isinstance(
        elapsed_time_s, (int, float)
    ):
        raise TypeError("elapsed_time_s must be a Python int or float")

    try:
        elapsed = float(elapsed_time_s)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "elapsed_time_s must be finite and nonnegative"
        ) from exc

    if not math.isfinite(elapsed):
        raise ValueError("elapsed_time_s must be finite")
    if elapsed < 0.0:
        raise ValueError("elapsed_time_s must be nonnegative")

    return elapsed


def _validate_extraction_geometry(
    generated_world: GeneratedWorld,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Validate authoritative generated-world extraction geometry."""

    center = generated_world.extraction_center
    radius = generated_world.extraction_radius_m

    if not isinstance(center, Tensor):
        raise TypeError(
            "generated_world.extraction_center must be a Tensor"
        )

    if tuple(center.shape) != (2,):
        raise ValueError(
            "generated_world.extraction_center must have shape (2,)"
        )

    if center.dtype != dtype:
        raise ValueError(
            "generated_world.extraction_center has the wrong dtype"
        )

    if center.device != device:
        raise ValueError(
            "generated_world.extraction_center is on the wrong device"
        )

    if not bool(torch.isfinite(center).all().item()):
        raise ValueError(
            "generated_world.extraction_center must be finite"
        )

    if isinstance(radius, bool) or not isinstance(radius, (int, float)):
        raise TypeError(
            "generated_world.extraction_radius_m must be a real scalar"
        )

    try:
        radius_value = float(radius)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "generated_world.extraction_radius_m must be finite and positive"
        ) from exc

    if not math.isfinite(radius_value) or radius_value <= 0.0:
        raise ValueError(
            "generated_world.extraction_radius_m must be finite and positive"
        )

    try:
        represented_radius = torch.tensor(
            radius_value,
            dtype=dtype,
            device=device,
        )
    except (RuntimeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "extraction radius cannot be represented in physics dtype"
        ) from exc

    if not bool(torch.isfinite(represented_radius).item()):
        raise ValueError(
            "extraction radius must be finite in physics dtype"
        )

    if not bool((represented_radius > 0).item()):
        raise ValueError(
            "extraction radius must remain positive in physics dtype"
        )

    return center, represented_radius


def _validate_episode_config(
    episode_config: EpisodeConfig,
) -> None:
    if not isinstance(episode_config, EpisodeConfig):
        raise TypeError("episode_config must be an EpisodeConfig")

    episode_config.validate()


def detect_extraction(
    episode_state: AgentEpisodeState,
    resolution: CollisionResolution,
    *,
    elapsed_time_s: float | int,
    episode_config: EpisodeConfig,
) -> ExtractionDetectionResult:
    """Inspect current post-resolution extraction eligibility without mutation.

    Success requires an alive, nonterminal agent with neither boundary
    escape nor unresolved obstacle contact, whose post-resolution center
    is within the generated-world extraction radius and whose explicit
    elapsed time is within the inclusive episode deadline.

    Extraction uses center-to-center distance. Agent body radius is not added.

    Resolved obstacle contact alone does not block extraction. Energy is not
    an eligibility condition.

    The time comparison uses Python scalar precision and does not round
    elapsed time into the physics dtype.
    """

    _validate_episode_config(episode_config)
    elapsed = _validate_elapsed_time(elapsed_time_s)

    _validate_episode_and_resolution(
        episode_state,
        resolution,
    )

    physics = resolution.physics
    batch_shape = tuple(episode_state.batch_shape)
    mask_shape = batch_shape + (1,)

    center, radius = _validate_extraction_geometry(
        episode_state.generated_world,
        dtype=physics.position.dtype,
        device=physics.position.device,
    )

    eligible = (
        physics.alive
        & ~episode_state.embodiment.extracted
        & ~episode_state.embodiment.failed
        & ~resolution.boundary_escape
        & ~resolution.unresolved_contact
    )

    delta = physics.position - center.reshape(
        (1,) * len(batch_shape) + (2,)
    )

    distance_squared = (delta * delta).sum(
        dim=-1,
        keepdim=True,
    )

    within_radius = distance_squared <= radius.square()

    deadline = float(episode_config.episode_duration_s)
    within_deadline = 0.0 <= elapsed <= deadline

    within_time = torch.full(
        mask_shape,
        within_deadline,
        dtype=torch.bool,
        device=physics.position.device,
    )

    success = eligible & within_radius & within_time

    return ExtractionDetectionResult(
        eligible=eligible.clone(),
        within_radius=within_radius.clone(),
        within_time=within_time,
        success=success.clone(),
        generated_world=episode_state.generated_world,
        elapsed_time_s=elapsed,
    )


def apply_extraction(
    episode_state: AgentEpisodeState,
    resolution: CollisionResolution,
    *,
    elapsed_time_s: float | int,
    episode_config: EpisodeConfig,
) -> ExtractionApplicationResult:
    """Recompute and apply current extraction successes.

    Unlike a cached detection result, application always evaluates the
    supplied current episode and collision report using the explicitly
    supplied current time and episode configuration.

    Consequently, an earlier successful detection cannot authorize
    extraction after the agent moves outside the extraction radius, after
    a disqualifying collision report, or after the episode deadline.

    Post-resolution physical state is installed through replace_physics.
    The current episode energy is preserved exactly, even when the
    collision snapshot contains older energy from before core collection.

    Successful transitions are recorded exclusively through mark_extracted.
    That accepted lifecycle method sets alive=False and preserves action
    history, measurements, objective bookkeeping, and other lifecycle state.
    """

    if not isinstance(episode_state, AgentEpisodeState):
        raise TypeError("episode_state must be an AgentEpisodeState")

    if not isinstance(resolution, CollisionResolution):
        raise TypeError("resolution must be a CollisionResolution")

    # Recompute from the current inputs. No previously returned detection
    # object or caller-provided mask participates in application.
    detection = detect_extraction(
        episode_state,
        resolution,
        elapsed_time_s=elapsed_time_s,
        episode_config=episode_config,
    )

    current_physics = episode_state.physics

    # Start with current post-resolution physical values, but preserve
    # energy from the current episode rather than stale resolution.energy.
    next_physics = resolution.physics.clone()
    next_physics.energy = current_physics.energy.clone()

    next_episode = episode_state.replace_physics(next_physics)

    next_episode = next_episode.mark_extracted(
        detection.success
    )

    return ExtractionApplicationResult(
        episode_state=next_episode,
        detection=detection,
        newly_extracted=detection.success.clone(),
    )
