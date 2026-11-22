from dataclasses import replace
import pytest
import torch

from lawless.objectives.cores import (
    collect_cores,
    detect_core_contacts,
)
from lawless.physics.constants import PhysicsConfig
from lawless.world.collision import CollisionResolution
from lawless.world.config import (
    EmbodimentConfig,
    EnvironmentConfig,
    ObjectiveConfig,
)
from lawless.world.episode_state import AgentEpisodeState
from lawless.world.full_generator import generate_world


def _generate_real_layout(*, device="cpu", dtype=torch.float32):
    config = EnvironmentConfig(seed=12)
    config.world.obstacle_count = 0
    config.objectives.core_count = 2

    return generate_world(
        config,
        seed=12,
        spawn_count=3,
        device=device,
        dtype=dtype,
    )


def _real_episode_config():
    return (
        EmbodimentConfig(agent_radius_m=0.5),
        ObjectiveConfig(core_count=2),
        PhysicsConfig(initial_energy=80.0),
    )


def _real_episode_state(*, device="cpu", dtype=torch.float32):
    layout = _generate_real_layout(device=device, dtype=dtype)
    embodiment_config, objective_config, physics_config = (
        _real_episode_config()
    )

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
            return torch.zeros(mask_shape, dtype=torch.bool, device=device)
        return value.clone()

    return CollisionResolution(
        physics=physics,
        contact=mask_or_zeros(contact),
        unresolved_contact=mask_or_zeros(unresolved_contact),
        boundary_escape=mask_or_zeros(boundary_escape),
    )


def _all_agents_at_core(state, core_index=0):
    center = state.generated_world.core_positions[core_index]
    return center.reshape(1, 1, 2).expand(
        *state.batch_shape, 2
    ).clone()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_generated_layout_core_contact_and_collection(dtype):
    state, objective_config, physics_config = _real_episode_state(
        dtype=dtype
    )
    positions = _all_agents_at_core(state, 0)
    resolution = _resolution(state, positions=positions)

    detected = detect_core_contacts(
        state,
        resolution,
        objective_config=objective_config,
    )

    assert detected.contact.shape == (2, 2, 2)
    assert detected.contact.dtype == torch.bool
    assert detected.contact.device == state.physics.position.device
    assert detected.eligible.shape == (2, 2, 1)
    assert detected.eligible.all()
    assert detected.contact[..., 0].all()
    assert not detected.contact[..., 1].any()

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert result.newly_collected.shape == (2, 2, 2)
    assert result.newly_collected.dtype == torch.bool
    assert result.newly_collected[..., 0].all()
    assert not result.newly_collected[..., 1].any()

    assert result.episode_state.objectives.collected[..., 0].all()
    assert result.episode_state.physics.energy.shape == (2, 2, 1)
    assert torch.allclose(
        result.episode_state.physics.energy,
        torch.full((2, 2, 1), 100.0, dtype=dtype),
    )
    assert torch.allclose(
        result.energy_restored,
        torch.full((2, 2, 1), 20.0, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_collection_is_per_agent_and_repeat_contact_is_idempotent(dtype):
    state, objective_config, physics_config = _real_episode_state(
        dtype=dtype
    )
    resolution = _resolution(
        state,
        positions=_all_agents_at_core(state, 0),
    )

    first = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )
    second = collect_cores(
        first.episode_state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert first.newly_collected[..., 0].all()
    assert not second.newly_collected.any()
    assert torch.equal(
        second.episode_state.objectives.collected,
        first.episode_state.objectives.collected,
    )
    assert torch.equal(
        second.episode_state.physics.energy,
        first.episode_state.physics.energy,
    )
    assert torch.equal(
        second.energy_restored,
        torch.zeros_like(second.energy_restored),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_energy_restored_is_actual_capped_increase(dtype):
    state, objective_config, _ = _real_episode_state(dtype=dtype)
    physics_config = PhysicsConfig(
        initial_energy=80.0,
        max_energy=90.0,
    )
    resolution = _resolution(
        state,
        positions=_all_agents_at_core(state, 0),
    )

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert torch.allclose(
        result.episode_state.physics.energy,
        torch.full((2, 2, 1), 90.0, dtype=dtype),
    )
    assert torch.allclose(
        result.energy_restored,
        torch.full((2, 2, 1), 10.0, dtype=dtype),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_event_agents_keep_energy_exactly_without_global_clamp(dtype):
    state, objective_config, physics_config = _real_episode_state(
        dtype=dtype
    )

    # The generated spawn positions are valid and separated from the cores.
    resolution = _resolution(state)

    # Use distinct in-range energy values, including one below capacity.
    physics = state.physics.clone()
    physics.energy = torch.tensor(
        [[[0.0], [13.0]], [[27.0], [80.0]]],
        dtype=dtype,
    )
    state = state.replace_physics(physics)
    resolution = _resolution(state)

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert not result.newly_collected.any()
    assert torch.equal(result.episode_state.physics.energy, physics.energy)
    assert torch.equal(
        result.energy_restored,
        torch.zeros_like(physics.energy),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_alive_zero_energy_agent_remains_eligible(dtype):
    state, objective_config, physics_config = _real_episode_state(
        dtype=dtype
    )

    physics = state.physics.clone()
    physics.energy[0, 0, 0] = 0.0
    state = state.replace_physics(physics)

    resolution = _resolution(
        state,
        positions=_all_agents_at_core(state, 0),
    )

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert result.eligible[0, 0, 0]
    assert result.newly_collected[0, 0, 0]
    assert result.episode_state.physics.alive[0, 0, 0]
    assert result.episode_state.physics.energy[0, 0, 0] > 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dead_agent_cannot_collect(dtype):
    state, objective_config, physics_config = _real_episode_state(
        dtype=dtype
    )

    # Put the agent at the core while it is still alive.
    positions = _all_agents_at_core(state, 0)
    state = state.replace_physics(
        _resolution(state, positions=positions).physics
    )

    # Now make it dead without moving its frozen position.
    dead_physics = state.physics.clone()
    dead_physics.alive[0, 0, 0] = False
    state = state.replace_physics(dead_physics)

    # Resolution must preserve the dead agent's position and lifecycle state.
    resolution = _resolution(state)

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert not result.eligible[0, 0, 0]
    assert not result.newly_collected[0, 0, 0]
    assert not result.episode_state.objectives.collected[0, 0, 0]
    assert not result.episode_state.physics.alive[0, 0, 0]
    assert torch.equal(
        result.episode_state.physics.position[0, 0],
        state.physics.position[0, 0],
    )
    assert torch.equal(
        result.episode_state.physics.energy[0, 0],
        state.physics.energy[0, 0],
    )


@pytest.mark.parametrize("terminal_method", ["mark_failed", "mark_extracted"])
def test_terminal_agents_cannot_collect(terminal_method):
    state, objective_config, physics_config = _real_episode_state()

    # Move to the core before applying the terminal lifecycle transition.
    positions = _all_agents_at_core(state, 0)
    state = state.replace_physics(
        _resolution(state, positions=positions).physics
    )

    mask = torch.zeros((2, 2, 1), dtype=torch.bool)
    mask[0, 0, 0] = True
    state = getattr(state, terminal_method)(mask)

    # Do not move the now-terminal agent in the collision snapshot.
    resolution = _resolution(state)

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert not result.eligible[0, 0, 0]
    assert not result.newly_collected[0, 0, 0]
    assert not result.episode_state.objectives.collected[0, 0, 0]
    assert not result.episode_state.physics.alive[0, 0, 0]
    assert torch.equal(
        result.episode_state.physics.position[0, 0],
        state.physics.position[0, 0],
    )

    if terminal_method == "mark_failed":
        assert result.episode_state.embodiment.failed[0, 0, 0]
        assert not result.episode_state.embodiment.extracted[0, 0, 0]
    else:
        assert result.episode_state.embodiment.extracted[0, 0, 0]
        assert not result.episode_state.embodiment.failed[0, 0, 0]


def test_resolved_obstacle_contact_does_not_block_collection():
    state, objective_config, physics_config = _real_episode_state()
    resolved_contact = torch.ones((2, 2, 1), dtype=torch.bool)
    resolution = _resolution(
        state,
        positions=_all_agents_at_core(state, 0),
        contact=resolved_contact,
    )

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert result.eligible.all()
    assert result.newly_collected[..., 0].all()


@pytest.mark.parametrize("report_name", ["boundary_escape", "unresolved_contact"])
def test_boundary_or_unresolved_contact_blocks_collection(report_name):
    state, objective_config, physics_config = _real_episode_state()
    report = torch.zeros((2, 2, 1), dtype=torch.bool)
    report[0, 0, 0] = True

    resolution = _resolution(
        state,
        positions=_all_agents_at_core(state, 0),
        **{report_name: report},
    )

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert not result.eligible[0, 0, 0]
    assert not result.newly_collected[0, 0, 0]
    assert result.newly_collected[1, 0, 0]
    assert not result.episode_state.embodiment.failed[0, 0, 0]


def test_post_resolution_position_is_used_for_contact():
    state, objective_config, physics_config = _real_episode_state()
    positions = _all_agents_at_core(state, 0)

    # The episode's original spawn positions are not changed. Only the
    # CollisionResolution's post-resolution physics is placed at the core.
    resolution = _resolution(state, positions=positions)

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert result.newly_collected[..., 0].all()
    assert torch.equal(state.physics.position, _real_episode_state()[0].physics.position)
    assert torch.equal(
        result.episode_state.physics.position,
        positions,
    )


def test_collection_preserves_history_terminal_state_and_input_ownership():
    state, objective_config, physics_config = _real_episode_state()
    resolution = _resolution(
        state,
        positions=_all_agents_at_core(state, 0),
    )

    original_energy = state.physics.energy.clone()
    original_alive = state.physics.alive.clone()
    original_actions = state.embodiment.previous_actions.clone()
    original_acceleration = state.embodiment.measured_acceleration.clone()
    original_extracted = state.embodiment.extracted.clone()
    original_failed = state.embodiment.failed.clone()
    original_collected = state.objectives.collected.clone()
    original_resolution_position = resolution.physics.position.clone()

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert torch.equal(state.physics.energy, original_energy)
    assert torch.equal(state.physics.alive, original_alive)
    assert torch.equal(state.embodiment.previous_actions, original_actions)
    assert torch.equal(
        state.embodiment.measured_acceleration,
        original_acceleration,
    )
    assert torch.equal(state.embodiment.extracted, original_extracted)
    assert torch.equal(state.embodiment.failed, original_failed)
    assert torch.equal(state.objectives.collected, original_collected)
    assert torch.equal(
        resolution.physics.position,
        original_resolution_position,
    )

    assert torch.equal(
        result.episode_state.physics.alive,
        original_alive,
    )
    assert torch.equal(
        result.episode_state.embodiment.previous_actions,
        original_actions,
    )
    assert torch.equal(
        result.episode_state.embodiment.measured_acceleration,
        original_acceleration,
    )
    assert torch.equal(
        result.episode_state.embodiment.extracted,
        original_extracted,
    )
    assert torch.equal(
        result.episode_state.embodiment.failed,
        original_failed,
    )


def test_objective_config_core_count_must_match_generated_layout():
    state, _, _ = _real_episode_state()
    wrong_config = ObjectiveConfig(core_count=3)
    resolution = _resolution(state)

    with pytest.raises(ValueError, match="core_count"):
        detect_core_contacts(
            state,
            resolution,
            objective_config=wrong_config,
        )


@pytest.mark.parametrize(
    "bad_energy",
    [float("nan"), float("inf"), 0.0, -1.0],
)
def test_invalid_core_energy_rejected_before_arithmetic(bad_energy):
    state, _, physics_config = _real_episode_state()
    objective_config = ObjectiveConfig(
        core_count=2,
        core_energy_j=bad_energy,
    )
    resolution = _resolution(state)

    with pytest.raises((ValueError, TypeError)):
        collect_cores(
            state,
            resolution,
            objective_config=objective_config,
            physics_config=physics_config,
        )


def test_input_energy_above_capacity_is_rejected():
    state, objective_config, physics_config = _real_episode_state()
    physics = state.physics.clone()
    physics.energy[0, 0, 0] = 101.0
    state = state.replace_physics(physics)
    resolution = _resolution(state)

    with pytest.raises(ValueError, match="max_energy"):
        collect_cores(
            state,
            resolution,
            objective_config=objective_config,
            physics_config=physics_config,
        )


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("boundary_escape", torch.zeros((2, 2), dtype=torch.bool)),
        ("unresolved_contact", torch.zeros((2, 2), dtype=torch.float32)),
        ("contact", torch.zeros((2, 2), dtype=torch.bool)),
    ],
)
def test_collision_report_shape_and_dtype_are_strict(field, bad_value):
    state, objective_config, physics_config = _real_episode_state()
    resolution = _resolution(state)

    # CollisionResolution is frozen; create a new value with the malformed
    # report rather than mutating the accepted instance.
    malformed_resolution = replace(
        resolution,
        **{field: bad_value},
    )

    with pytest.raises((TypeError, ValueError)):
        collect_cores(
            state,
            malformed_resolution,
            objective_config=objective_config,
            physics_config=physics_config,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_core_layout_has_correct_empty_contact_shape(dtype):
    config = EnvironmentConfig(seed=12)
    config.world.obstacle_count = 0
    config.objectives.core_count = 0
    layout = generate_world(
        config,
        seed=12,
        spawn_count=3,
        device="cpu",
        dtype=dtype,
    )

    embodiment_config = EmbodimentConfig(agent_radius_m=0.5)
    objective_config = ObjectiveConfig(core_count=0)
    physics_config = PhysicsConfig(initial_energy=80.0)
    spawn_indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)

    state = AgentEpisodeState.initialize(
        layout,
        spawn_indices,
        embodiment_config=embodiment_config,
        objective_config=objective_config,
        physics_config=physics_config,
    )
    resolution = _resolution(state)

    detected = detect_core_contacts(
        state,
        resolution,
        objective_config=objective_config,
    )
    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert detected.contact.shape == (2, 2, 0)
    assert result.newly_collected.shape == (2, 2, 0)
    assert not result.newly_collected.any()
    assert torch.equal(result.episode_state.physics.energy, state.physics.energy)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires an actual CUDA device",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_core_collection_device_and_cpu_parity(dtype):
    cuda_device = torch.device("cuda", torch.cuda.current_device())

    cpu_state, cpu_objectives, cpu_physics = _real_episode_state(
        device="cpu",
        dtype=dtype,
    )
    cuda_state, cuda_objectives, cuda_physics = _real_episode_state(
        device=cuda_device,
        dtype=dtype,
    )

    cpu_positions = _all_agents_at_core(cpu_state, 0)
    cuda_positions = _all_agents_at_core(cuda_state, 0)

    cpu_resolution = _resolution(cpu_state, positions=cpu_positions)
    cuda_resolution = _resolution(cuda_state, positions=cuda_positions)

    cpu_result = collect_cores(
        cpu_state,
        cpu_resolution,
        objective_config=cpu_objectives,
        physics_config=cpu_physics,
    )
    cuda_result = collect_cores(
        cuda_state,
        cuda_resolution,
        objective_config=cuda_objectives,
        physics_config=cuda_physics,
    )

    assert cuda_result.contact.device == cuda_device
    assert cuda_result.eligible.device == cuda_device
    assert cuda_result.newly_collected.device == cuda_device
    assert cuda_result.energy_restored.device == cuda_device
    assert cuda_result.episode_state.physics.position.device == cuda_device
    assert cuda_result.episode_state.physics.energy.device == cuda_device
    assert cuda_result.episode_state.objectives.collected.device == cuda_device

    assert cuda_result.contact.dtype == torch.bool
    assert cuda_result.newly_collected.dtype == torch.bool
    assert cuda_result.episode_state.physics.energy.dtype == dtype

    assert torch.equal(
        cuda_result.contact.cpu(),
        cpu_result.contact,
    )
    assert torch.equal(
        cuda_result.newly_collected.cpu(),
        cpu_result.newly_collected,
    )
    torch.testing.assert_close(
        cuda_result.episode_state.physics.energy.cpu(),
        cpu_result.episode_state.physics.energy,
        rtol=1e-6 if dtype == torch.float32 else 1e-12,
        atol=1e-6 if dtype == torch.float32 else 1e-12,
    )
    torch.testing.assert_close(
        cuda_result.energy_restored.cpu(),
        cpu_result.energy_restored,
        rtol=1e-6 if dtype == torch.float32 else 1e-12,
        atol=1e-6 if dtype == torch.float32 else 1e-12,
    )


@pytest.mark.parametrize("source", ["episode", "resolution"])
def test_non_tensor_physics_field_is_rejected_before_shape_validation(source):
    state, objective_config, physics_config = _real_episode_state()
    resolution = _resolution(state)

    if source == "episode":
        # AgentState is mutable; the containing episode state remains intact.
        state.physics.position = None
        candidate_state = state
        candidate_resolution = resolution
    else:
        malformed_physics = resolution.physics.clone()
        malformed_physics.position = None
        candidate_state = state
        candidate_resolution = replace(
            resolution,
            physics=malformed_physics,
        )

    with pytest.raises(TypeError, match="position"):
        collect_cores(
            candidate_state,
            candidate_resolution,
            objective_config=objective_config,
            physics_config=physics_config,
        )


def test_resolution_energy_snapshot_need_not_equal_current_episode_energy():
    state, objective_config, physics_config = _real_episode_state()

    # Construct a collision snapshot with deliberately stale energy.
    resolution = _resolution(
        state,
        positions=_all_agents_at_core(state, 0),
    )
    resolution_physics = resolution.physics.clone()
    resolution_physics.energy.fill_(1.0)
    resolution = replace(
        resolution,
        physics=resolution_physics,
    )

    # The episode state is authoritative for energy arithmetic.
    current_physics = state.physics.clone()
    current_physics.energy.fill_(80.0)
    state = state.replace_physics(current_physics)

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert torch.allclose(
        result.episode_state.physics.energy,
        torch.full((2, 2, 1), 100.0),
    )
    assert torch.allclose(
        result.energy_restored,
        torch.full((2, 2, 1), 20.0),
    )


from lawless.world.collision import resolve_static_obstacles
from lawless.world.generated_world import GeneratedWorld
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState

def _controlled_episode(
    *,
    dtype=torch.float32,
    initial_energy=20.0,
    obstacle_centers=(),
    obstacle_radii=(),
    core_centers=((40.0, 50.0), (42.0, 50.0)),
    core_radii=(0.75, 0.75),
):
    """Build a constructor-valid 100 m square controlled episode."""
    device = torch.device("cpu")

    obstacle_centers_tensor = torch.tensor(
        obstacle_centers,
        dtype=dtype,
        device=device,
    ).reshape(-1, 2)
    obstacle_radii_tensor = torch.tensor(
        obstacle_radii,
        dtype=dtype,
        device=device,
    ).reshape(-1)

    obstacles = CircleObstacles(
        centers=obstacle_centers_tensor,
        radii=obstacle_radii_tensor,
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
            [80.0, 80.0],
            dtype=dtype,
            device=device,
        ),
        extraction_radius_m=3.0,
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

    state = AgentEpisodeState.initialize(
        layout,
        torch.tensor([0], dtype=torch.int64, device=device),
        embodiment_config=embodiment_config,
        objective_config=objective_config,
        physics_config=physics_config,
    )
    return state, objective_config, physics_config


def _move_alive_agent(state, position):
    """Move the sole live agent using the accepted replacement API."""
    physics = state.physics.clone()
    physics.position[0] = torch.as_tensor(
        position,
        dtype=physics.position.dtype,
        device=physics.position.device,
    )
    return state.replace_physics(physics)


def _resolve_controlled_episode(state):
    """Run the actual P1.3 endpoint resolver on the episode physics."""
    return resolve_static_obstacles(
        state.physics,
        state.generated_world.world,
        agent_radius_m=state.generated_world.agent_body_radius_m,
    )

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_smallest_supported_episode_batch_shape_one(dtype):
    state, objective_config, physics_config = _controlled_episode(
        dtype=dtype,
    )

    resolution = _resolution(state)
    detected = detect_core_contacts(
        state,
        resolution,
        objective_config=objective_config,
    )
    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert state.batch_shape == (1,)
    assert state.physics.position.shape == (1, 2)
    assert state.physics.energy.shape == (1, 1)
    assert detected.contact.shape == (1, 2)
    assert detected.eligible.shape == (1, 1)
    assert result.newly_collected.shape == (1, 2)
    assert result.energy_restored.shape == (1, 1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_spawn_indices_are_rejected_by_episode_initialization(dtype):
    state, objective_config, physics_config = _controlled_episode(
        dtype=dtype,
    )
    layout = state.generated_world

    with pytest.raises(ValueError, match="at least one batch dimension"):
        AgentEpisodeState.initialize(
            layout,
            torch.tensor(0, dtype=torch.int64),
            embodiment_config=EmbodimentConfig(agent_radius_m=0.5),
            objective_config=objective_config,
            physics_config=physics_config,
        )

@pytest.mark.parametrize("source", ["episode", "resolution"])
@pytest.mark.parametrize(
    "field",
    [
        "position",
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
    ],
)
def test_nonfinite_physics_fields_are_rejected(source, field):
    state, objective_config, physics_config = _real_episode_state()
    resolution = _resolution(state)

    if source == "episode":
        physics = state.physics
    else:
        physics = resolution.physics

    # Deliberately corrupt the selected owned tensor so the objective
    # validator, rather than an earlier replacement API, sees the NaN.
    selected_tensor = getattr(physics, field)
    selected_tensor.reshape(-1)[0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        collect_cores(
            state,
            resolution,
            objective_config=objective_config,
            physics_config=physics_config,
        )

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "initial_energy,expected_energy,expected_restored",
    [
        (20.0, 70.0, 50.0),
        (80.0, 100.0, 20.0),
    ],
)
def test_simultaneous_two_core_collection_after_real_resolution(
    dtype,
    initial_energy,
    expected_energy,
    expected_restored,
):
    state, objective_config, physics_config = _controlled_episode(
        dtype=dtype,
        initial_energy=initial_energy,
        obstacle_centers=(),
        obstacle_radii=(),
        core_centers=((40.0, 50.0), (42.0, 50.0)),
        core_radii=(0.75, 0.75),
    )
    state = _move_alive_agent(state, (41.0, 50.0))

    resolution = _resolve_controlled_episode(state)
    assert not resolution.boundary_escape.any()
    assert not resolution.unresolved_contact.any()

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert result.contact.shape == (1, 2)
    assert result.contact.all()
    assert result.newly_collected.shape == (1, 2)
    assert result.newly_collected.all()
    assert result.episode_state.objectives.collected.all()
    torch.testing.assert_close(
        result.episode_state.physics.energy,
        torch.tensor([[expected_energy]], dtype=dtype),
    )
    torch.testing.assert_close(
        result.energy_restored,
        torch.tensor([[expected_restored]], dtype=dtype),
    )
    assert result.episode_state.physics.alive.all()

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_core_contact_includes_exact_tangency_but_not_just_outside(dtype):
    state, objective_config, _ = _controlled_episode(
        dtype=dtype,
        core_centers=((40.0, 50.0), (42.0, 50.0)),
        core_radii=(0.75, 0.75),
    )

    tangent_x = torch.tensor(41.25, dtype=dtype)
    outside_x = torch.nextafter(
        tangent_x,
        torch.tensor(float("inf"), dtype=dtype),
    )

    tangent_resolution = _resolution(
        state,
        positions=torch.tensor(
            [[tangent_x.item(), 50.0]],
            dtype=dtype,
        ),
    )
    outside_resolution = _resolution(
        state,
        positions=torch.tensor(
            [[outside_x.item(), 50.0]],
            dtype=dtype,
        ),
    )

    tangent = detect_core_contacts(
        state,
        tangent_resolution,
        objective_config=objective_config,
    )
    outside = detect_core_contacts(
        state,
        outside_resolution,
        objective_config=objective_config,
    )

    assert tangent.contact[0, 0]
    assert not outside.contact[0, 0]

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_heterogeneous_generated_core_radii_control_contact(dtype):
    state, objective_config, _ = _controlled_episode(
        dtype=dtype,
        core_centers=((40.0, 50.0), (44.0, 50.0)),
        core_radii=(0.5, 1.25),
    )

    assert objective_config.core_radius_m == 0.75
    assert torch.equal(
        state.generated_world.core_radii,
        torch.tensor([0.5, 1.25], dtype=dtype),
    )

    # Distance to core 1 is 1.5: within 0.5 + 1.25.
    # Distance to core 0 is 5.5: outside 0.5 + 0.5.
    resolution = _resolution(
        state,
        positions=torch.tensor([[45.5, 50.0]], dtype=dtype),
    )
    detected = detect_core_contacts(
        state,
        resolution,
        objective_config=objective_config,
    )

    assert detected.contact.tolist() == [[False, True]]

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_resolver_boundary_escape_blocks_core_collection(dtype):
    state, objective_config, physics_config = _controlled_episode(
        dtype=dtype,
        obstacle_centers=(),
        obstacle_radii=(),
        core_centers=((0.75, 50.0), (42.0, 50.0)),
        core_radii=(0.75, 0.75),
    )
    state = _move_alive_agent(state, (-0.25, 50.0))

    resolution = _resolve_controlled_episode(state)

    assert resolution.boundary_escape[0, 0]
    assert not resolution.unresolved_contact[0, 0]

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    # Geometrically, the agent is within 0.25 + 0.75 of the first core.
    # The resolver's boundary report nevertheless makes it ineligible.
    assert not result.eligible[0, 0]
    assert not result.contact[0, 0]
    assert not result.newly_collected[0, 0]
    assert not result.episode_state.objectives.collected[0, 0]
    assert not result.episode_state.embodiment.failed[0, 0]

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_resolver_obstacle_correction_can_collect_nearby_core(dtype):
    state, objective_config, physics_config = _controlled_episode(
        dtype=dtype,
        obstacle_centers=((50.0, 50.0),),
        obstacle_radii=(5.0,),
        core_centers=((56.0, 50.0), (42.0, 70.0)),
        core_radii=(0.75, 0.75),
    )

    # This point is inside the obstacle's expanded radius of 5.5 m.
    state = _move_alive_agent(state, (54.5, 50.0))
    original_position = state.physics.position.clone()

    resolution = _resolve_controlled_episode(state)

    assert resolution.contact[0, 0]
    assert not resolution.unresolved_contact[0, 0]
    assert not resolution.boundary_escape[0, 0]
    assert not torch.equal(resolution.physics.position, original_position)

    # The resolver should place the agent on the obstacle's expanded
    # boundary, at which point the nearby core is still in contact.
    expected_resolved_position = torch.tensor(
        [[55.5, 50.0]],
        dtype=dtype,
    )
    torch.testing.assert_close(
        resolution.physics.position,
        expected_resolved_position,
        rtol=1e-6 if dtype == torch.float32 else 1e-12,
        atol=1e-6 if dtype == torch.float32 else 1e-12,
    )

    result = collect_cores(
        state,
        resolution,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    assert result.eligible[0, 0]
    assert result.contact[0, 0]
    assert result.newly_collected[0, 0]
    assert result.episode_state.objectives.collected[0, 0]
    assert result.episode_state.physics.alive[0, 0]
    torch.testing.assert_close(
        result.episode_state.physics.position,
        resolution.physics.position,
    )
    torch.testing.assert_close(
        result.episode_state.physics.energy,
        torch.tensor([[45.0]], dtype=dtype),
    )
    torch.testing.assert_close(
        result.energy_restored,
        torch.tensor([[25.0]], dtype=dtype),
    )
