"""Energy-core contact detection and one-time collection."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lawless.physics.constants import PhysicsConfig
from lawless.physics.state import AgentState
from lawless.world.collision import CollisionResolution
from lawless.world.config import ObjectiveConfig
from lawless.world.episode_state import AgentEpisodeState

from lawless.objectives._validation import (
    _energy_parameters,
    _validate_episode_and_resolution,
    _validate_input_energy,
)


@dataclass(frozen=True)
class CoreContactResult:
    """Per-agent masks describing eligible, currently collectible core contact."""

    contact: Tensor
    eligible: Tensor


@dataclass(frozen=True)
class CoreCollectionResult:
    """Updated episode and masks/amounts describing this collection operation."""

    episode_state: AgentEpisodeState
    contact: Tensor
    eligible: Tensor
    newly_collected: Tensor
    energy_restored: Tensor


def detect_core_contacts(
    episode_state: AgentEpisodeState,
    resolution: CollisionResolution,
    *,
    objective_config: ObjectiveConfig | None = None,
) -> CoreContactResult:
    """Detect eligible core contacts using post-resolution physical positions.

    Core locations and individual core radii come exclusively from
    ``episode_state.generated_world``. If supplied, ``objective_config`` is
    used to validate the configured core count; it does not override geometry.

    Contact is inclusive circle-to-circle contact:
    distance(agent center, core center) <= agent radius + core radius.

    An agent is eligible only when alive, not extracted, not failed, and
    neither boundary-escaped nor left in unresolved obstacle contact.
    Resolved obstacle contact alone does not block collection.

    Energy is not consulted here. In particular, alive=True with energy=0
    remains eligible under the accepted P0 lifecycle contract.
    """
    _validate_episode_and_resolution(
        episode_state,
        resolution,
        objective_config=objective_config,
    )

    world = episode_state.generated_world
    physics = resolution.physics
    batch_shape = tuple(episode_state.batch_shape)
    device = physics.position.device
    dtype = physics.position.dtype

    alive = physics.alive
    extracted = episode_state.embodiment.extracted
    failed = episode_state.embodiment.failed

    collision_clear = (
        ~resolution.boundary_escape
        & ~resolution.unresolved_contact
    )

    eligible = alive & ~extracted & ~failed & collision_clear

    core_count = int(world.core_positions.shape[0])
    collected = episode_state.objectives.collected

    if core_count == 0:
        contact = torch.zeros(
            batch_shape + (0,),
            dtype=torch.bool,
            device=device,
        )
        return CoreContactResult(contact=contact, eligible=eligible)

    # Use the represented, validated geometry from GeneratedWorld.
    agent_radius = torch.tensor(
        world.agent_body_radius_m,
        dtype=dtype,
        device=device,
    )
    combined_radii = agent_radius + world.core_radii

    delta = (
        physics.position.unsqueeze(-2)
        - world.core_positions.reshape(
            (1,) * len(batch_shape) + (core_count, 2)
        )
    )
    distance_squared = (delta * delta).sum(dim=-1)
    radius_squared = combined_radii.square()

    geometric_contact = distance_squared <= radius_squared

    contact = (
        geometric_contact
        & eligible
        & ~collected
    )

    return CoreContactResult(
        contact=contact,
        eligible=eligible,
    )


def collect_cores(
    episode_state: AgentEpisodeState,
    resolution: CollisionResolution,
    *,
    objective_config: ObjectiveConfig,
    physics_config: PhysicsConfig,
) -> CoreCollectionResult:
    """Collect each newly contacted core once per agent and restore energy.

    The operation uses the post-resolution ``resolution.physics`` as its
    physical-state basis. It records events through
    ``AgentEpisodeState.mark_collected`` and installs the resulting physics
    through ``AgentEpisodeState.replace_physics``.

    Collection never changes ``alive``, ``extracted``, ``failed``,
    action history, or measured acceleration.

    Input energy must be finite and within [0, physics_config.max_energy].
    Core energy and maximum energy must be finite, positive, and representable
    as positive finite values in the physics dtype.

    Energy restoration is awarded only for newly collected cores:

        next_energy = min(max_energy, old_energy + new_count * core_energy)

    Agents with no new collection event retain their input energy exactly;
    no global clamp is applied to no-event or ineligible agents.

    ``energy_restored`` is the actual capped increase
    ``next_energy - old_energy``, not the nominal core reward.
    """
    _validate_episode_and_resolution(
        episode_state,
        resolution,
        objective_config=objective_config,
    )

    physics = resolution.physics
    batch_shape = tuple(episode_state.batch_shape)
    dtype = physics.position.dtype
    device = physics.position.device

    core_energy, max_energy = _energy_parameters(
        objective_config,
        physics_config,
        dtype=dtype,
        device=device,
    )

    input_energy = episode_state.physics.energy
    _validate_input_energy(input_energy, max_energy=max_energy)

    detected = detect_core_contacts(
        episode_state,
        resolution,
        objective_config=objective_config,
    )
    contact = detected.contact

    core_count = int(episode_state.generated_world.core_positions.shape[0])
    agent_count = 1
    for dimension in batch_shape:
        agent_count *= dimension

    flat_contact = contact.reshape(agent_count, core_count)
    agent_indices, core_indices = flat_contact.nonzero(as_tuple=True)

    # The existing state API owns duplicate handling, validation, and
    # independent per-agent collection bookkeeping.
    next_episode, event_newly_collected = episode_state.mark_collected(
        agent_indices,
        core_indices,
    )

    flat_newly_collected = torch.zeros_like(flat_contact)
    if event_newly_collected.numel() > 0:
        flat_newly_collected[agent_indices, core_indices] = (
            event_newly_collected
        )

    newly_collected = flat_newly_collected.reshape(
        batch_shape + (core_count,)
    )

    new_count = newly_collected.sum(
        dim=-1,
        keepdim=True,
    ).to(dtype=dtype)

    has_new_collection = new_count > 0

    # Use the episode's current energy, not a potentially unrelated energy
    # tensor supplied in a separately constructed collision result.
    candidate_energy = torch.minimum(
        input_energy + new_count * core_energy,
        max_energy,
    )

    # Preserve energy bit-for-bit for agents with no newly collected core.
    next_energy = torch.where(
        has_new_collection,
        candidate_energy,
        input_energy,
    )

    energy_restored = torch.where(
        has_new_collection,
        next_energy - input_energy,
        torch.zeros_like(input_energy),
    )

    # Start from post-resolution physics, clone it, and replace only energy.
    # The accepted episode API preserves action history and lifecycle state.
    next_physics = physics.clone()
    next_physics.energy = next_energy

    next_episode = next_episode.replace_physics(next_physics)

    return CoreCollectionResult(
        episode_state=next_episode,
        contact=contact,
        eligible=detected.eligible,
        newly_collected=newly_collected,
        energy_restored=energy_restored,
    )
