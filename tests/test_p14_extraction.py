"""P1.4 extraction detection and application regressions."""

from __future__ import annotations

import math

import pytest
import torch

from lawless.objectives import (
    ExtractionApplicationResult,
    ExtractionDetectionResult,
    apply_extraction,
    collect_cores,
    detect_extraction,
)
from lawless.physics.constants import PhysicsConfig
from lawless.world.collision import CollisionResolution, resolve_static_obstacles
from lawless.world.config import (
    EmbodimentConfig,
    EpisodeConfig,
    ObjectiveConfig,
)
from lawless.world.episode_state import AgentEpisodeState
from lawless.world.generated_world import GeneratedWorld
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState
from lawless.world.config import EnvironmentConfig
from lawless.world.full_generator import generate_world


DTYPES = (torch.float32, torch.float64)


def _episode_config(duration=40.0):
    return EpisodeConfig(
        episode_duration_s=duration,
        shift_time_s=duration / 2,
        dt_s=1.0 / 60.0,
    )


def _controlled_episode(
    *,
    dtype=torch.float32,
    initial_energy=20.0,
    obstacle_centers=(),
    obstacle_radii=(),
    core_centers=((40.0, 50.0), (42.0, 50.0)),
    core_radii=(0.75, 0.75),
    extraction_center=(80.0, 80.0),
    extraction_radius=3.0,
    batch_shape=(1,),
):
    """Build a constructor-valid controlled 100 m square episode."""

    device = torch.device("cpu")

    obstacles = CircleObstacles(
        centers=torch.tensor(
            obstacle_centers,
            dtype=dtype,
            device=device,
        ).reshape(-1, 2),
        radii=torch.tensor(
            obstacle_radii,
            dtype=dtype,
            device=device,
        ).reshape(-1),
    )

    world = WorldState(
        width_m=100.0,
        height_m=100.0,
        obstacles=obstacles,
    )

    layout = GeneratedWorld(
        world=world,
        spawn_positions=torch.tensor(
            [[10.0, 10.0]],
            dtype=dtype,
            device=device,
        ),
        agent_body_radius_m=0.5,
        extraction_center=torch.tensor(
            extraction_center,
            dtype=dtype,
            device=device,
        ),
        extraction_radius_m=extraction_radius,
        core_positions=torch.tensor(
            core_centers,
            dtype=dtype,
            device=device,
        ).reshape(-1, 2),
        core_radii=torch.tensor(
            core_radii,
            dtype=dtype,
            device=device,
        ),
        seed=12,
    )

    embodiment_config = EmbodimentConfig(agent_radius_m=0.5)
    objective_config = ObjectiveConfig(
        core_count=len(core_radii),
    )
    physics_config = PhysicsConfig(
        initial_energy=initial_energy,
        max_energy=100.0,
    )

    spawn_indices = torch.zeros(
        batch_shape,
        dtype=torch.int64,
        device=device,
    )

    state = AgentEpisodeState.initialize(
        layout,
        spawn_indices,
        embodiment_config=embodiment_config,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    return state, objective_config, physics_config


def _real_episode(*, device="cpu", dtype=torch.float32):
    config = EnvironmentConfig(seed=12)
    config.world.obstacle_count = 0
    config.objectives.core_count = 2

    layout = generate_world(
        config,
        seed=12,
        spawn_count=3,
        device=device,
        dtype=dtype,
    )

    embodiment_config = EmbodimentConfig(agent_radius_m=0.5)
    objective_config = ObjectiveConfig(core_count=2)
    physics_config = PhysicsConfig(initial_energy=80.0)

    spawn_indices = torch.tensor(
        [[0, 1], [1, 2]],
        dtype=torch.int64,
        device=device,
    )

    state = AgentEpisodeState.initialize(
        layout,
        spawn_indices,
        embodiment_config=embodiment_config,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    return state, objective_config, physics_config


def _resolution(
    state,
    *,
    positions=None,
    contact=None,
    unresolved_contact=None,
    boundary_escape=None,
):
    physics = state.physics.clone()
    batch_shape = tuple(state.batch_shape)
    mask_shape = batch_shape + (1,)
    device = physics.position.device

    if positions is not None:
        assert positions.shape == physics.position.shape
        physics.position = positions.clone()

    def mask_or_zeros(value):
        if value is None:
            return torch.zeros(
                mask_shape,
                dtype=torch.bool,
                device=device,
            )
        return value.clone()

    return CollisionResolution(
        physics=physics,
        contact=mask_or_zeros(contact),
        unresolved_contact=mask_or_zeros(unresolved_contact),
        boundary_escape=mask_or_zeros(boundary_escape),
    )


def _move_single_agent(state, position):
    physics = state.physics.clone()
    physics.position[0] = torch.tensor(
        position,
        dtype=physics.position.dtype,
        device=physics.position.device,
    )
    return state.replace_physics(physics)


def _at_extraction(state):
    center = state.generated_world.extraction_center
    return center.reshape(
        (1,) * len(state.batch_shape) + (2,)
    ).expand(*state.batch_shape, 2).clone()


def _detect(state, resolution, elapsed=10.0, config=None):
    return detect_extraction(
        state,
        resolution,
        elapsed_time_s=elapsed,
        episode_config=config or _episode_config(),
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_inside_extraction_radius_succeeds(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (81.0, 80.0))
    resolution = _resolution(state)

    result = _detect(state, resolution)

    assert result.success.tolist() == [[True]]
    assert result.eligible.tolist() == [[True]]
    assert result.within_radius.tolist() == [[True]]
    assert result.within_time.tolist() == [[True]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_extraction_radius_tangency_is_inclusive(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (83.0, 80.0))
    resolution = _resolution(state)

    result = _detect(state, resolution)

    assert result.success.tolist() == [[True]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_next_representable_position_outside_radius_fails(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)

    outside = torch.nextafter(
        torch.tensor(83.0, dtype=dtype),
        torch.tensor(float("inf"), dtype=dtype),
    ).item()

    state = _move_single_agent(state, (outside, 80.0))
    result = _detect(state, _resolution(state))

    assert result.within_radius.tolist() == [[False]]
    assert result.success.tolist() == [[False]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_extraction_does_not_add_agent_body_radius(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)

    # Distance 3.25 is outside the extraction radius 3.0,
    # although agent radius 0.5 would make the expanded radius 3.5.
    state = _move_single_agent(state, (83.25, 80.0))

    result = _detect(state, _resolution(state))

    assert result.within_radius.tolist() == [[False]]
    assert result.success.tolist() == [[False]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_deadline_is_inclusive(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (80.0, 80.0))

    result = _detect(
        state,
        _resolution(state),
        elapsed=40.0,
        config=_episode_config(40.0),
    )

    assert result.within_time.tolist() == [[True]]
    assert result.success.tolist() == [[True]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_next_representable_time_after_deadline_fails(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (80.0, 80.0))

    just_after = math.nextafter(40.0, math.inf)

    result = _detect(
        state,
        _resolution(state),
        elapsed=just_after,
        config=_episode_config(40.0),
    )

    assert result.within_radius.tolist() == [[True]]
    assert result.within_time.tolist() == [[False]]
    assert result.success.tolist() == [[False]]
    assert result.elapsed_time_s == just_after


@pytest.mark.parametrize(
    "elapsed,error",
    [
        (True, TypeError),
        (False, TypeError),
        ("10", TypeError),
        (None, TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (-float("inf"), ValueError),
        (-0.001, ValueError),
    ],
)
def test_invalid_elapsed_times_are_rejected(elapsed, error):
    state, _, _ = _controlled_episode()

    with pytest.raises(error):
        _detect(state, _resolution(state), elapsed=elapsed)


def test_zero_elapsed_time_is_valid():
    state, _, _ = _controlled_episode()
    state = _move_single_agent(state, (80.0, 80.0))

    result = _detect(
        state,
        _resolution(state),
        elapsed=0,
    )

    assert result.success.tolist() == [[True]]


def test_invalid_episode_config_type_is_rejected():
    state, _, _ = _controlled_episode()

    with pytest.raises(TypeError, match="EpisodeConfig"):
        detect_extraction(
            state,
            _resolution(state),
            elapsed_time_s=1.0,
            episode_config=object(),
        )


@pytest.mark.parametrize(
    "config",
    [
        EpisodeConfig(
            episode_duration_s=0.0,
            shift_time_s=0.0,
        ),
        EpisodeConfig(
            episode_duration_s=40.0,
            shift_time_s=40.0,
        ),
        EpisodeConfig(
            episode_duration_s=float("nan"),
            shift_time_s=0.0,
        ),
    ],
)
def test_invalid_episode_config_values_are_rejected(config):
    state, _, _ = _controlled_episode()

    with pytest.raises((TypeError, ValueError)):
        detect_extraction(
            state,
            _resolution(state),
            elapsed_time_s=1.0,
            episode_config=config,
        )



@pytest.mark.parametrize("dtype", DTYPES)
def test_dead_agent_cannot_extract(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)

    # Move while alive, then transition to dead through the accepted
    # replace_physics contract. Do not modify frozen position afterward.
    state = _move_single_agent(state, (80.0, 80.0))

    physics = state.physics.clone()
    physics.alive[0, 0] = False
    state = state.replace_physics(physics)

    assert state.physics.energy.item() == pytest.approx(20.0)
    assert state.embodiment.extracted.tolist() == [[False]]
    assert state.embodiment.failed.tolist() == [[False]]

    result = _detect(state, _resolution(state))

    assert result.eligible.tolist() == [[False]]
    assert result.success.tolist() == [[False]]



@pytest.mark.parametrize("dtype", DTYPES)
def test_extracted_agent_cannot_extract_again(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (80.0, 80.0))

    first_resolution = _resolution(state)
    first = apply_extraction(
        state,
        first_resolution,
        elapsed_time_s=10.0,
        episode_config=_episode_config(),
    )

    assert first.newly_extracted.tolist() == [[True]]
    assert first.episode_state.embodiment.extracted.tolist() == [[True]]
    assert first.episode_state.physics.alive.tolist() == [[False]]

    fresh_resolution = _resolution(first.episode_state)

    repeated = apply_extraction(
        first.episode_state,
        fresh_resolution,
        elapsed_time_s=11.0,
        episode_config=_episode_config(),
    )

    assert repeated.newly_extracted.tolist() == [[False]]
    assert repeated.detection.eligible.tolist() == [[False]]
    assert repeated.episode_state.embodiment.extracted.tolist() == [[True]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_failed_agent_cannot_extract(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = state.mark_failed(
        torch.tensor([[True]], dtype=torch.bool)
    )

    result = _detect(state, _resolution(state))

    assert result.eligible.tolist() == [[False]]
    assert result.success.tolist() == [[False]]
    assert state.embodiment.failed.tolist() == [[True]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_alive_agent_with_zero_energy_can_extract(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)

    physics = state.physics.clone()
    physics.energy.zero_()
    physics.position[0] = torch.tensor(
        [80.0, 80.0],
        dtype=dtype,
    )
    state = state.replace_physics(physics)

    result = _detect(state, _resolution(state))

    assert result.eligible.tolist() == [[True]]
    assert result.success.tolist() == [[True]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_extraction_does_not_require_core_collection(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (80.0, 80.0))

    assert not bool(state.objectives.collected.any().item())

    result = _detect(state, _resolution(state))

    assert result.success.tolist() == [[True]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_resolved_obstacle_contact_does_not_block_extraction(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (80.0, 80.0))

    contact = torch.tensor([[True]], dtype=torch.bool)

    resolution = _resolution(
        state,
        contact=contact,
    )

    result = _detect(state, resolution)

    assert result.eligible.tolist() == [[True]]
    assert result.success.tolist() == [[True]]


@pytest.mark.parametrize(
    "report_field",
    ["boundary_escape", "unresolved_contact"],
)
def test_disqualifying_collision_reports_block_extraction(report_field):
    state, _, _ = _controlled_episode()
    state = _move_single_agent(state, (80.0, 80.0))

    report = torch.tensor([[True]], dtype=torch.bool)

    resolution = _resolution(
        state,
        **{report_field: report},
    )

    result = _detect(state, resolution)

    assert result.eligible.tolist() == [[False]]
    assert result.within_radius.tolist() == [[True]]
    assert result.success.tolist() == [[False]]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("contact", torch.zeros((1,), dtype=torch.bool), ValueError),
        ("unresolved_contact", torch.zeros((1, 1), dtype=torch.float32), TypeError),
        ("boundary_escape", torch.zeros((1, 1), dtype=torch.int64), TypeError),
    ],
)
def test_invalid_collision_report_shape_and_dtype_are_rejected(
    field,
    value,
    error,
):
    state, _, _ = _controlled_episode()

    physics = state.physics.clone()

    reports = {
        "contact": torch.zeros((1, 1), dtype=torch.bool),
        "unresolved_contact": torch.zeros((1, 1), dtype=torch.bool),
        "boundary_escape": torch.zeros((1, 1), dtype=torch.bool),
    }
    reports[field] = value

    resolution = CollisionResolution(
        physics=physics,
        **reports,
    )

    with pytest.raises(error):
        _detect(state, resolution)



@pytest.mark.parametrize("dtype", DTYPES)
def test_application_preserves_objectives_history_measurements_and_energy(dtype):
    state, _, _ = _controlled_episode(
        dtype=dtype,
        initial_energy=37.0,
    )
    state = _move_single_agent(state, (80.0, 80.0))

    # Accepted transition-measurement layout is batch_shape + (2,).
    actions = torch.tensor(
        [[0.25, -0.5]],
        dtype=dtype,
    )
    acceleration = torch.tensor(
        [[1.25, -2.5]],
        dtype=dtype,
    )

    state = state.record_transition_measurements(
        applied_actions=actions,
        measured_acceleration=acceleration,
    )

    original_energy = state.physics.energy.clone()
    original_actions = state.embodiment.previous_actions.clone()
    original_measurements = (
        state.embodiment.measured_acceleration.clone()
    )
    original_collected = state.objectives.collected.clone()
    original_position = state.physics.position.clone()
    original_spawn_indices = state.spawn_indices.clone()

    resolution = _resolution(state)

    application = apply_extraction(
        state,
        resolution,
        elapsed_time_s=10.0,
        episode_config=_episode_config(),
    )

    next_state = application.episode_state

    assert isinstance(application, ExtractionApplicationResult)
    assert application.newly_extracted.tolist() == [[True]]
    assert application.detection.success.tolist() == [[True]]

    assert torch.equal(next_state.physics.position, original_position)
    assert torch.equal(next_state.physics.energy, original_energy)

    assert torch.equal(
        next_state.embodiment.previous_actions,
        original_actions,
    )
    assert torch.equal(
        next_state.embodiment.measured_acceleration,
        original_measurements,
    )
    assert torch.equal(
        next_state.objectives.collected,
        original_collected,
    )
    assert torch.equal(
        next_state.spawn_indices,
        original_spawn_indices,
    )

    assert next_state.physics.alive.tolist() == [[False]]
    assert next_state.embodiment.extracted.tolist() == [[True]]
    assert next_state.embodiment.failed.tolist() == [[False]]
    assert next_state.generated_world is state.generated_world



@pytest.mark.parametrize("dtype", DTYPES)
def test_application_does_not_mutate_input_episode_or_collision_snapshot(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (80.0, 80.0))

    resolution = _resolution(state)

    original_alive = state.physics.alive.clone()
    original_energy = state.physics.energy.clone()
    original_position = resolution.physics.position.clone()
    original_resolution_energy = resolution.physics.energy.clone()

    application = apply_extraction(
        state,
        resolution,
        elapsed_time_s=10.0,
        episode_config=_episode_config(),
    )

    assert application.episode_state is not state
    assert torch.equal(state.physics.alive, original_alive)
    assert torch.equal(state.physics.energy, original_energy)

    assert torch.equal(resolution.physics.position, original_position)
    assert torch.equal(
        resolution.physics.energy,
        original_resolution_energy,
    )

    assert (
        application.episode_state.physics.position.data_ptr()
        != resolution.physics.position.data_ptr()
    )







@pytest.mark.parametrize("dtype", DTYPES)
def test_sequential_collection_then_extraction_preserves_restored_energy(dtype):
    state, objective_config, physics_config = _controlled_episode(
        dtype=dtype,
        initial_energy=20.0,
        core_centers=((84.0, 80.0), (42.0, 50.0)),
        core_radii=(0.75, 0.75),
    )

    state = _move_single_agent(state, (83.0, 80.0))

    # One real collision snapshot is intentionally reused.
    resolution = resolve_static_obstacles(
        state.physics,
        state.generated_world.world,
        agent_radius_m=state.generated_world.agent_body_radius_m,
    )

    collection = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    collected_state = collection.episode_state

    assert collection.newly_collected.tolist() == [[True, False]]
    assert collected_state.physics.energy.item() == pytest.approx(45.0)

    application = apply_extraction(
        collected_state,
        resolution,
        elapsed_time_s=10.0,
        episode_config=_episode_config(),
    )

    next_state = application.episode_state

    assert application.detection.success.tolist() == [[True]]
    assert next_state.physics.energy.item() == pytest.approx(45.0)
    assert resolution.physics.energy.item() == pytest.approx(20.0)

    assert next_state.objectives.collected.tolist() == [[True, False]]
    assert next_state.embodiment.extracted.tolist() == [[True]]
    assert next_state.physics.alive.tolist() == [[False]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_real_generated_layout_detection(dtype):
    state, _, _ = _real_episode(dtype=dtype)

    resolution = _resolution(
        state,
        positions=_at_extraction(state),
    )

    result = _detect(state, resolution)

    assert result.success.shape == state.batch_shape + (1,)
    assert result.success.dtype == torch.bool
    assert result.success.device == state.physics.position.device
    assert bool(result.success.all().item())


@pytest.mark.parametrize("dtype", DTYPES)
def test_real_p13_boundary_resolver_report_blocks_extraction(dtype):
    state, _, _ = _controlled_episode(
        dtype=dtype,
        obstacle_centers=((4.0, 50.0),),
        obstacle_radii=(4.0,),
    )

    # Put the agent near the left boundary and away from extraction.
    state = _move_single_agent(state, (0.1, 50.0))

    resolution = resolve_static_obstacles(
        state.physics,
        state.generated_world.world,
        agent_radius_m=state.generated_world.agent_body_radius_m,
    )

    assert resolution.boundary_escape.shape == (1, 1)

    result = _detect(state, resolution)

    assert not bool(result.success.any().item())
    assert not bool(result.eligible.any().item())




@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA hardware is unavailable",
)
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_extraction_has_indexed_device_and_cpu_parity(dtype):
    cpu_state, _, _ = _real_episode(
        device="cpu",
        dtype=dtype,
    )

    cuda_state, _, _ = _real_episode(
        device="cuda:0",
        dtype=dtype,
    )

    assert cuda_state.physics.position.device == torch.device("cuda:0")

    cpu_resolution = _resolution(
        cpu_state,
        positions=_at_extraction(cpu_state),
    )
    cuda_resolution = _resolution(
        cuda_state,
        positions=_at_extraction(cuda_state),
    )

    cpu_result = _detect(cpu_state, cpu_resolution)
    cuda_result = _detect(cuda_state, cuda_resolution)

    for mask_name in (
        "eligible",
        "within_radius",
        "within_time",
        "success",
    ):
        cpu_mask = getattr(cpu_result, mask_name)
        cuda_mask = getattr(cuda_result, mask_name)

        assert cpu_mask.device == torch.device("cpu")
        assert cuda_mask.device == torch.device("cuda:0")
        assert torch.equal(cpu_mask, cuda_mask.cpu())

    cuda_application = apply_extraction(
        cuda_state,
        cuda_resolution,
        elapsed_time_s=10.0,
        episode_config=_episode_config(),
    )

    assert (
        cuda_application.episode_state.physics.position.device
        == torch.device("cuda:0")
    )
    assert (
        cuda_application.episode_state.physics.energy.device
        == torch.device("cuda:0")
    )
    assert cuda_application.newly_extracted.device == torch.device("cuda:0")

    assert torch.equal(
        cuda_application.episode_state.physics.alive.cpu(),
        torch.zeros_like(cpu_state.physics.alive),
    )



@pytest.mark.parametrize("dtype", DTYPES)
def test_application_rechecks_geometry_after_agent_moves_outside(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (80.0, 80.0))

    old_resolution = _resolution(state)
    old_detection = _detect(
        state,
        old_resolution,
        elapsed=1.0,
    )

    assert old_detection.success.tolist() == [[True]]

    # Move the still-alive agent outside before applying extraction.
    outside_state = _move_single_agent(state, (70.0, 80.0))
    current_resolution = _resolution(outside_state)

    application = apply_extraction(
        outside_state,
        current_resolution,
        elapsed_time_s=1.0,
        episode_config=_episode_config(),
    )

    assert application.detection.within_radius.tolist() == [[False]]
    assert application.detection.success.tolist() == [[False]]
    assert application.newly_extracted.tolist() == [[False]]
    assert application.episode_state.embodiment.extracted.tolist() == [[False]]
    assert application.episode_state.physics.alive.tolist() == [[True]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_application_rechecks_current_collision_disqualification(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (80.0, 80.0))

    original_resolution = _resolution(state)
    original_detection = _detect(
        state,
        original_resolution,
        elapsed=1.0,
    )

    assert original_detection.success.tolist() == [[True]]

    # Same valid physical position, but the current resolver report now
    # identifies an unresolved contact.
    current_resolution = _resolution(
        state,
        unresolved_contact=torch.tensor(
            [[True]],
            dtype=torch.bool,
        ),
    )

    application = apply_extraction(
        state,
        current_resolution,
        elapsed_time_s=1.0,
        episode_config=_episode_config(),
    )

    assert application.detection.within_radius.tolist() == [[True]]
    assert application.detection.eligible.tolist() == [[False]]
    assert application.detection.success.tolist() == [[False]]
    assert application.newly_extracted.tolist() == [[False]]
    assert application.episode_state.physics.alive.tolist() == [[True]]


@pytest.mark.parametrize("dtype", DTYPES)
def test_application_rechecks_current_deadline(dtype):
    state, _, _ = _controlled_episode(dtype=dtype)
    state = _move_single_agent(state, (80.0, 80.0))

    resolution = _resolution(state)

    old_detection = _detect(
        state,
        resolution,
        elapsed=40.0,
        config=_episode_config(40.0),
    )

    assert old_detection.success.tolist() == [[True]]

    just_after_deadline = math.nextafter(40.0, math.inf)

    application = apply_extraction(
        state,
        resolution,
        elapsed_time_s=just_after_deadline,
        episode_config=_episode_config(40.0),
    )

    assert application.detection.within_radius.tolist() == [[True]]
    assert application.detection.within_time.tolist() == [[False]]
    assert application.detection.success.tolist() == [[False]]
    assert application.newly_extracted.tolist() == [[False]]
    assert application.episode_state.embodiment.extracted.tolist() == [[False]]
    assert application.episode_state.physics.alive.tolist() == [[True]]



@pytest.mark.parametrize("dtype", DTYPES)
def test_real_p13_obstacle_resolution_enters_extraction_region(dtype):
    state, _, _ = _controlled_episode(
        dtype=dtype,
        obstacle_centers=((75.75, 80.0),),
        obstacle_radii=(1.0,),
        extraction_center=(80.0, 80.0),
        extraction_radius=3.0,
    )

    state = _move_single_agent(state, (76.0, 80.0))

    resolution = resolve_static_obstacles(
        state.physics,
        state.generated_world.world,
        agent_radius_m=state.generated_world.agent_body_radius_m,
    )

    expected_position = torch.tensor(
        [77.25, 80.0],
        dtype=dtype,
        device=state.physics.position.device,
    )

    assert torch.allclose(
        resolution.physics.position[0],
        expected_position,
        atol=1e-5,
        rtol=0.0,
    )

    assert resolution.contact.tolist() == [[True]]
    assert resolution.boundary_escape.tolist() == [[False]]
    assert resolution.unresolved_contact.tolist() == [[False]]

    # The original position is outside the extraction radius:
    # distance = 4.0. The resolved position is inside:
    # distance = 2.75.
    assert torch.linalg.vector_norm(
        state.physics.position[0]
        - state.generated_world.extraction_center
    ).item() > 3.0

    application = apply_extraction(
        state,
        resolution,
        elapsed_time_s=10.0,
        episode_config=_episode_config(),
    )

    assert application.detection.within_radius.tolist() == [[True]]
    assert application.detection.eligible.tolist() == [[True]]
    assert application.detection.success.tolist() == [[True]]
    assert application.newly_extracted.tolist() == [[True]]
    assert application.episode_state.embodiment.extracted.tolist() == [[True]]
    assert application.episode_state.physics.alive.tolist() == [[False]]
