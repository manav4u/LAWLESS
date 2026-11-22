"""Atomic, per-world stepping for the LAWLESS P1 environment."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from lawless.environment.episode import EnvironmentState, WorldEpisodeState
from lawless.objectives.cores import collect_cores
from lawless.objectives.extraction import apply_extraction
from lawless.physics.dynamics import rotate_gravity_90
from lawless.physics.simulator import step as physics_step
from lawless.sensors.lidar import sense_lidar
from lawless.sensors.observations import assemble_observations
from lawless.world.collision import resolve_static_obstacles


def _observe(world: WorldEpisodeState, sensor_config) -> Tensor:
    lidar = sense_lidar(
        world.episode.physics,
        world.episode.generated_world.world,
        sensor_config,
    )
    return assemble_observations(world.episode, lidar)


def _validate_actions(actions: Tensor, environment: EnvironmentState) -> Tensor:
    if not isinstance(actions, Tensor):
        raise TypeError("actions must be a torch.Tensor")
    expected = environment.batch_shape + (2,)
    if tuple(actions.shape) != expected:
        raise ValueError(f"actions must have shape {expected}, got {tuple(actions.shape)}")
    if actions.is_complex():
        raise TypeError("actions must be real-valued")
    if not bool(torch.isfinite(actions).all().item()):
        raise ValueError("actions must contain only finite values")

    reference = environment.worlds[0].episode.physics.position
    converted = actions.to(device=reference.device, dtype=reference.dtype)
    # P0's established action contract clips finite commands after conversion.
    return converted.clamp(-1.0, 1.0)


def _zero_world_info(world: WorldEpisodeState, core_count: int) -> dict[str, Tensor]:
    physics = world.episode.physics
    n = world.num_agents
    device, dtype = physics.position.device, physics.position.dtype
    zmask = torch.zeros((n, 1), dtype=torch.bool, device=device)
    return {
        "newly_collected": torch.zeros((n, core_count), dtype=torch.bool, device=device),
        "energy_restored": torch.zeros((n, 1), dtype=dtype, device=device),
        "energy_spent": torch.zeros((n, 1), dtype=dtype, device=device),
        "newly_extracted": zmask.clone(),
        "collision_contact": zmask.clone(),
        "unresolved_contact": zmask.clone(),
        "boundary_escape": zmask.clone(),
        "failure_this_step": zmask.clone(),
        "energy_death_this_step": zmask.clone(),
        "actuators_enabled": zmask.clone(),
        "shift_applied_this_step": torch.tensor(False, dtype=torch.bool, device=device),
        "gravity_used": world.gravity.clone(),
        "gravity_next": world.gravity.clone(),
        "elapsed_steps": torch.tensor(world.step_count, dtype=torch.int64, device=device),
        "elapsed_time_s": torch.tensor(
            world.step_count * world.physics_config.dt,
            dtype=torch.float64,
            device=device,
        ),
        "world_completed": torch.tensor(world.completed, dtype=torch.bool, device=device),
    }


def _advance_world(
    world: WorldEpisodeState,
    actions: Tensor,
    environment: EnvironmentState,
) -> tuple[WorldEpisodeState, dict[str, Tensor]]:
    config = environment.config
    episode = world.episode
    before = episode.physics
    dt = world.physics_config.dt
    next_step = world.step_count + 1
    episode_steps = int(round(
        config.episode.episode_duration_s / config.episode.dt_s
    ))
    elapsed = (
        float(config.episode.episode_duration_s)
        if next_step == episode_steps
        else next_step * dt
    )

    physics_next, p0_diagnostics = physics_step(
        before,
        actions,
        world.gravity,
        world.physics_config,
    )
    resolution = resolve_static_obstacles(
        physics_next,
        episode.generated_world.world,
        agent_radius_m=episode.generated_world.agent_body_radius_m,
    )

    # Install the collision-resolved physical state and record measured
    # acceleration from the actual velocity change, not P0 force diagnostics.
    next_episode = episode.replace_physics(resolution.physics)
    measured_acceleration = (
        resolution.physics.velocity - before.velocity
    ) / dt
    next_episode = next_episode.record_transition_measurements(
        applied_actions=actions,
        measured_acceleration=measured_acceleration,
    )

    core_result = collect_cores(
        next_episode,
        resolution,
        objective_config=config.objectives,
        physics_config=world.physics_config,
    )
    extraction_result = apply_extraction(
        core_result.episode_state,
        resolution,
        elapsed_time_s=elapsed,
        episode_config=config.episode,
    )
    next_episode = extraction_result.episode_state

    # Collision failure is an annotation only for agents that remain
    # physically alive. A P0 energy death stays dead and unlabelled.
    failure_requested = (
        resolution.boundary_escape | resolution.unresolved_contact
    ) & next_episode.physics.alive
    next_episode = next_episode.mark_failed(failure_requested)

    shift_steps = int(round(
        config.episode.shift_time_s / config.episode.dt_s
    ))
    at_deadline = next_step >= episode_steps

    truncated = world.truncated.clone()
    if at_deadline:
        # Extraction at the inclusive deadline has already been applied.
        truncated = truncated | next_episode.physics.alive

    completed = bool(
        ((~next_episode.physics.alive) | truncated).all().item()
    )

    gravity_used = world.gravity.clone()
    gravity_next = world.gravity.clone()
    shift_this_step = next_step == shift_steps and not world.shift_applied
    shift_applied = world.shift_applied
    if shift_this_step:
        gravity_next = rotate_gravity_90(
            world.gravity,
            world.shift_direction,
        )
        shift_applied = True

    next_world = WorldEpisodeState(
        episode=next_episode,
        physics_config=world.physics_config,
        gravity=gravity_next,
        initial_gravity=world.initial_gravity,
        shift_direction=world.shift_direction,
        step_count=next_step,
        shift_applied=shift_applied,
        truncated=truncated,
        completed=completed,
        world_seed=world.world_seed,
        condition_seed=world.condition_seed,
        linear_damping_s_inv=world.linear_damping_s_inv,
    )

    info = {
        "newly_collected": core_result.newly_collected,
        "energy_restored": core_result.energy_restored,
        "energy_spent": p0_diagnostics["energy_spent"],
        "newly_extracted": extraction_result.newly_extracted,
        "collision_contact": resolution.contact,
        "unresolved_contact": resolution.unresolved_contact,
        "boundary_escape": resolution.boundary_escape,
        "failure_this_step": failure_requested,
        "energy_death_this_step": before.alive & ~physics_next.alive,
        "actuators_enabled": p0_diagnostics["actuators_enabled"],
        "shift_applied_this_step": torch.tensor(
            shift_this_step, dtype=torch.bool, device=before.position.device
        ),
        "gravity_used": gravity_used,
        "gravity_next": gravity_next.clone(),
        "elapsed_steps": torch.tensor(
            next_step, dtype=torch.int64, device=before.position.device
        ),
        "elapsed_time_s": torch.tensor(
            elapsed, dtype=torch.float64, device=before.position.device
        ),
        "world_completed": torch.tensor(
            completed, dtype=torch.bool, device=before.position.device
        ),
    }
    return next_world, info


def step(
    environment: EnvironmentState,
    actions: Tensor,
) -> tuple[EnvironmentState, Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
    """Advance active worlds once, returning observations and raw outcomes.

    Args:
        environment: State returned by :func:`reset` or a prior :func:`step`.
        actions: Finite tensor with exact shape [B, N, 2]. Finite values
            outside [-1, 1] are accepted and clipped by the P0 contract.

    Returns:
        ``(next_environment, observations, rewards, terminated, truncated,
        info)``. Rewards are the raw newly-extracted success indicator only;
        they are not a shaped reward or final V1 fitness score.

    A completed world is frozen while other worlds continue. Its physical
    state, observations, terminated mask, and truncated mask remain stable.
    Once every world is complete, another call raises without mutation.
    """
    if not isinstance(environment, EnvironmentState):
        raise TypeError("environment must be an EnvironmentState")

    # Validate the complete action tensor before advancing any world.
    applied_actions = _validate_actions(actions, environment)
    if environment.all_completed:
        raise RuntimeError("all environment worlds are complete; reset before stepping")

    next_worlds: list[WorldEpisodeState] = []
    per_world_info: list[dict[str, Tensor]] = []
    for world, world_actions in zip(environment.worlds, applied_actions):
        if world.completed:
            next_worlds.append(world)
            per_world_info.append(
                _zero_world_info(
                    world,
                    int(world.episode.generated_world.core_positions.shape[0]),
                )
            )
        else:
            next_world, info = _advance_world(
                world,
                world_actions,
                environment,
            )
            next_worlds.append(next_world)
            per_world_info.append(info)

    next_environment = EnvironmentState(
        worlds=tuple(next_worlds),
        config=environment.config,
        num_agents=environment.num_agents,
    )
    observations = torch.stack(
        [_observe(world, environment.config.sensors) for world in next_worlds],
        dim=0,
    )
    rewards = torch.stack(
        [
            info["newly_extracted"].squeeze(-1).to(
                dtype=world.episode.physics.position.dtype
            )
            for info, world in zip(per_world_info, next_worlds)
        ],
        dim=0,
    )
    terminated = torch.stack(
        [world.terminated for world in next_worlds],
        dim=0,
    )
    truncated = torch.stack(
        [world.truncated.squeeze(-1) for world in next_worlds],
        dim=0,
    )

    info = {
        "newly_collected": torch.stack(
            [item["newly_collected"] for item in per_world_info], dim=0
        ),
        "core_collection_count": torch.stack(
            [item["newly_collected"].sum(dim=-1) for item in per_world_info],
            dim=0,
        ),
        "energy_restored": torch.stack(
            [item["energy_restored"].squeeze(-1) for item in per_world_info],
            dim=0,
        ),
        "energy_spent": torch.stack(
            [item["energy_spent"].squeeze(-1) for item in per_world_info],
            dim=0,
        ),
        "newly_extracted": torch.stack(
            [item["newly_extracted"].squeeze(-1) for item in per_world_info],
            dim=0,
        ),
        "collision_contact": torch.stack(
            [item["collision_contact"].squeeze(-1) for item in per_world_info],
            dim=0,
        ),
        "unresolved_contact": torch.stack(
            [item["unresolved_contact"].squeeze(-1) for item in per_world_info],
            dim=0,
        ),
        "boundary_escape": torch.stack(
            [item["boundary_escape"].squeeze(-1) for item in per_world_info],
            dim=0,
        ),
        "failure_this_step": torch.stack(
            [item["failure_this_step"].squeeze(-1) for item in per_world_info],
            dim=0,
        ),
        "energy_death_this_step": torch.stack(
            [item["energy_death_this_step"].squeeze(-1) for item in per_world_info],
            dim=0,
        ),
        "actuators_enabled": torch.stack(
            [item["actuators_enabled"].squeeze(-1) for item in per_world_info],
            dim=0,
        ),
        "gravity_used": torch.stack(
            [item["gravity_used"] for item in per_world_info], dim=0
        ),
        "gravity_next": torch.stack(
            [item["gravity_next"] for item in per_world_info], dim=0
        ),
        "shift_applied_this_step": torch.stack(
            [item["shift_applied_this_step"] for item in per_world_info], dim=0
        ),
        "shift_applied": torch.tensor(
            [world.shift_applied for world in next_worlds],
            dtype=torch.bool,
            device=observations.device,
        ),
        "elapsed_steps": torch.stack(
            [item["elapsed_steps"] for item in per_world_info], dim=0
        ),
        "elapsed_time_s": torch.stack(
            [item["elapsed_time_s"] for item in per_world_info], dim=0
        ),
        "world_completed": torch.stack(
            [item["world_completed"] for item in per_world_info], dim=0
        ),
        "physical_alive": torch.stack(
            [world.episode.physics.alive.squeeze(-1) for world in next_worlds],
            dim=0,
        ),
        "extracted": torch.stack(
            [world.episode.embodiment.extracted.squeeze(-1) for world in next_worlds],
            dim=0,
        ),
        "failed": torch.stack(
            [world.episode.embodiment.failed.squeeze(-1) for world in next_worlds],
            dim=0,
        ),
        "truncated": truncated.clone(),
        "terminated": terminated.clone(),
    }
    return next_environment, observations, rewards, terminated, truncated, info
