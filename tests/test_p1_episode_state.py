import pytest
import torch

from lawless.physics.constants import PhysicsConfig
from lawless.physics.state import AgentState
from lawless.world.config import EmbodimentConfig, ObjectiveConfig
from lawless.world.embodiment import AgentEmbodimentState
from lawless.world.episode_state import AgentEpisodeState
from lawless.world.generated_world import GeneratedWorld
from lawless.world.objective_state import AgentCoreCollectionState


def _layout(dtype=torch.float32, device="cpu"):
    """Minimal layout-shaped test instance; geometry validation is tested elsewhere."""
    device = torch.device(device)

    layout = object.__new__(GeneratedWorld)
    object.__setattr__(layout, "world", object())
    object.__setattr__(
        layout,
        "spawn_positions",
        torch.tensor(
            [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
            dtype=dtype,
            device=device,
        ),
    )
    represented_radius = torch.tensor(0.5, dtype=dtype).item()
    object.__setattr__(layout, "agent_body_radius_m", represented_radius)
    object.__setattr__(
        layout,
        "extraction_center",
        torch.tensor([50.0, 50.0], dtype=dtype, device=device),
    )
    object.__setattr__(layout, "extraction_radius_m", 3.0)
    object.__setattr__(
        layout,
        "core_positions",
        torch.tensor(
            [[15.0, 15.0], [25.0, 25.0]],
            dtype=dtype,
            device=device,
        ),
    )
    object.__setattr__(
        layout,
        "core_radii",
        torch.tensor([0.75, 0.75], dtype=dtype, device=device),
    )
    object.__setattr__(layout, "seed", 42)
    return layout


def _initialize(
    *,
    device="cpu",
    dtype=torch.float32,
    spawn_indices=None,
    energy=100.0,
    radius=0.5,
):
    layout = _layout(dtype=dtype, device=device)
    if spawn_indices is None:
        spawn_indices = torch.tensor(
            [[0, 1], [2, 0]],
            dtype=torch.int64,
            device=device,
        )
    return AgentEpisodeState.initialize(
        layout,
        spawn_indices,
        embodiment_config=EmbodimentConfig(agent_radius_m=radius),
        objective_config=ObjectiveConfig(core_count=2),
        physics_config=PhysicsConfig(initial_energy=energy),
    )


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda",
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
)])
def test_episode_initialization_uses_explicit_spawn_assignments(device):
    state = _initialize(device=device)
    expected = torch.tensor(
        [
            [[10.0, 10.0], [20.0, 20.0]],
            [[30.0, 30.0], [10.0, 10.0]],
        ],
        dtype=torch.float32,
    )

    assert state.physics.position.device.type == torch.device(device).type
    assert torch.equal(state.physics.position.cpu(), expected)
    assert state.batch_shape == (2, 2)
    assert state.objectives.collected.shape == (2, 2, 2)
    assert not bool(state.objectives.collected.any().item())
    assert bool(state.physics.alive.all().item())
    assert torch.equal(state.embodiment.previous_actions, torch.zeros_like(
        state.embodiment.previous_actions
    ))
    assert torch.equal(
        state.embodiment.measured_acceleration,
        torch.zeros_like(state.embodiment.measured_acceleration),
    )
    assert not bool(state.embodiment.extracted.any().item())
    assert not bool(state.embodiment.failed.any().item())


def test_initial_measurement_buffer_means_no_prior_measurement():
    state = _initialize()
    assert torch.equal(
        state.embodiment.measured_acceleration,
        torch.zeros_like(state.embodiment.measured_acceleration),
    )
    # The integration stores no fabricated measurement-validity flag:
    # callers must supply a real measurement after a completed transition.


def test_zero_initial_energy_does_not_infer_failure_or_death():
    state = _initialize(energy=0.0)
    assert bool(state.physics.alive.all().item())
    assert not bool(state.embodiment.failed.any().item())
    assert not bool(state.embodiment.extracted.any().item())


def test_layout_is_shared_by_reference_and_geometry_is_not_mutated():
    state = _initialize()
    layout = state.generated_world
    original_spawns = layout.spawn_positions.clone()
    original_cores = layout.core_positions.clone()

    clone = state.clone()

    assert state.generated_world is layout
    assert clone.generated_world is layout
    assert torch.equal(layout.spawn_positions, original_spawns)
    assert torch.equal(layout.core_positions, original_cores)


def test_clone_owns_all_per_agent_tensor_storage():
    state = _initialize()
    clone = state.clone()

    assert clone.generated_world is state.generated_world
    assert clone.spawn_indices.data_ptr() != state.spawn_indices.data_ptr()
    assert clone.physics.position.data_ptr() != state.physics.position.data_ptr()
    assert (
        clone.embodiment.previous_actions.data_ptr()
        != state.embodiment.previous_actions.data_ptr()
    )
    assert (
        clone.embodiment.measured_acceleration.data_ptr()
        != state.embodiment.measured_acceleration.data_ptr()
    )
    assert clone.embodiment.extracted.data_ptr() != state.embodiment.extracted.data_ptr()
    assert clone.embodiment.failed.data_ptr() != state.embodiment.failed.data_ptr()
    assert (
        clone.objectives.collected.data_ptr()
        != state.objectives.collected.data_ptr()
    )


def test_mutating_clone_tensors_does_not_leak_to_original():
    state = _initialize()
    clone = state.clone()

    clone.physics.position[0, 0, 0] = -99
    clone.embodiment.previous_actions[0, 0, 0] = 0.75
    clone.embodiment.measured_acceleration[0, 0, 0] = 12.0
    clone.objectives.collected[0, 0, 0] = True
    clone.spawn_indices[0, 0] = 2

    assert state.physics.position[0, 0, 0] != -99
    assert state.embodiment.previous_actions[0, 0, 0] == 0
    assert state.embodiment.measured_acceleration[0, 0, 0] == 0
    assert not bool(state.objectives.collected[0, 0, 0].item())
    assert state.spawn_indices[0, 0] == 0


def test_reset_creates_fresh_state_and_requires_new_assignments():
    state = _initialize()
    collected, _ = state.mark_collected(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )
    measured = collected.record_transition_measurements(
        applied_actions=torch.full((2, 2, 2), 0.5),
        measured_acceleration=torch.full((2, 2, 2), 3.0),
    )
    terminal = measured.mark_failed(
        torch.tensor(
            [[[True], [False]], [[False], [False]]],
            dtype=torch.bool,
        )
    )

    new_assignments = torch.tensor([[2, 1], [0, 2]], dtype=torch.int64)
    reset = AgentEpisodeState.reset(
        state.generated_world,
        new_assignments,
        embodiment_config=EmbodimentConfig(agent_radius_m=0.5),
        objective_config=ObjectiveConfig(core_count=2),
        physics_config=PhysicsConfig(initial_energy=75.0),
    )

    assert reset.generated_world is state.generated_world
    assert torch.equal(reset.spawn_indices, new_assignments)
    assert torch.equal(
        reset.physics.position,
        state.generated_world.spawn_positions.index_select(
            0, new_assignments.reshape(-1)
        ).reshape(2, 2, 2),
    )
    assert torch.equal(reset.physics.energy, torch.full((2, 2, 1), 75.0))
    assert bool(reset.physics.alive.all().item())
    assert not bool(reset.embodiment.failed.any().item())
    assert not bool(reset.embodiment.extracted.any().item())
    assert not bool(reset.objectives.collected.any().item())
    assert not bool(reset.embodiment.previous_actions.any().item())
    assert not bool(reset.embodiment.measured_acceleration.any().item())

    # Reset does not mutate the preceding episode.
    assert bool(terminal.embodiment.failed[0, 0, 0].item())
    assert bool(terminal.objectives.collected[0, 0, 1].item())
    assert torch.equal(
        terminal.embodiment.measured_acceleration,
        torch.full((2, 2, 2), 3.0),
    )


def test_radius_must_match_layout_dtype_representation():
    layout = _layout(dtype=torch.float32)
    represented = torch.tensor(0.5, dtype=torch.float32).item()

    accepted = AgentEpisodeState.initialize(
        layout,
        torch.tensor([0], dtype=torch.int64),
        embodiment_config=EmbodimentConfig(agent_radius_m=represented),
        objective_config=ObjectiveConfig(core_count=2),
        physics_config=PhysicsConfig(),
    )
    assert accepted.physics.position.dtype == torch.float32

    with pytest.raises(ValueError, match="represented generated-world"):
        AgentEpisodeState.initialize(
            layout,
            torch.tensor([0], dtype=torch.int64),
            embodiment_config=EmbodimentConfig(agent_radius_m=0.500001),
            objective_config=ObjectiveConfig(core_count=2),
            physics_config=PhysicsConfig(),
        )


@pytest.mark.parametrize(
    ("indices", "error", "message"),
    [
        (
            torch.tensor([0, 1], dtype=torch.int32),
            TypeError,
            "int64",
        ),
    ],
)
def test_spawn_assignment_validation(indices, error, message):
    layout = _layout()
    with pytest.raises(error, match=message):
        AgentEpisodeState.initialize(
            layout,
            indices,
            embodiment_config=EmbodimentConfig(agent_radius_m=0.5),
            objective_config=ObjectiveConfig(core_count=2),
            physics_config=PhysicsConfig(),
        )


@pytest.mark.parametrize(
    ("indices", "expected_shape", "expected_positions"),
    [
        (
            torch.tensor([0], dtype=torch.int64),
            (1,),
            [[10.0, 10.0]],
        ),
        (
            torch.tensor([[0, 2]], dtype=torch.int64),
            (1, 2),
            [[[10.0, 10.0], [30.0, 30.0]]],
        ),
        (
            torch.tensor([[2], [1]], dtype=torch.int64),
            (2, 1),
            [[[30.0, 30.0]], [[20.0, 20.0]]],
        ),
    ],
)
def test_spawn_assignment_shape_defines_batch_shape_without_broadcasting(
    indices,
    expected_shape,
    expected_positions,
):
    state = _initialize(spawn_indices=indices)

    assert state.batch_shape == expected_shape
    assert tuple(state.spawn_indices.shape) == expected_shape
    assert tuple(state.physics.position.shape) == expected_shape + (2,)
    assert tuple(state.physics.energy.shape) == expected_shape + (1,)
    assert tuple(state.objectives.collected.shape) == expected_shape + (2,)
    assert torch.equal(
        state.physics.position,
        torch.tensor(expected_positions, dtype=state.physics.position.dtype),
    )


def test_spawn_assignment_device_must_match_layout():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    layout = _layout(device="cuda")
    with pytest.raises(ValueError, match="must be on"):
        AgentEpisodeState.initialize(
            layout,
            torch.tensor([0], dtype=torch.int64, device="cpu"),
            embodiment_config=EmbodimentConfig(agent_radius_m=0.5),
            objective_config=ObjectiveConfig(core_count=2),
            physics_config=PhysicsConfig(),
        )


def test_objective_events_are_independent_per_agent_and_preserved_by_physics_replace():
    state = _initialize()
    collected, newly = state.mark_collected(
        torch.tensor([0, 1, 0], dtype=torch.int64),
        torch.tensor([1, 1, 1], dtype=torch.int64),
    )

    assert newly.tolist() == [True, True, False]
    assert bool(collected.objectives.collected[0, 0, 1].item())
    assert bool(collected.objectives.collected[0, 1, 1].item())

    replacement = collected.physics.clone()
    replacement.position[0, 0] += 1
    replaced = collected.replace_physics(replacement)

    assert torch.equal(
        replaced.objectives.collected,
        collected.objectives.collected,
    )
    assert replaced.generated_world is collected.generated_world
    assert replaced.objectives.collected.data_ptr() != (
        collected.objectives.collected.data_ptr()
    )


def test_physics_replacement_preserves_history_and_lifecycle():
    state = _initialize()
    measured = state.record_transition_measurements(
        applied_actions=torch.full((2, 2, 2), 0.25),
        measured_acceleration=torch.full((2, 2, 2), 2.5),
    )
    terminal = measured.mark_extracted(
        torch.tensor(
            [[[True], [False]], [[False], [False]]],
            dtype=torch.bool,
        )
    )

    replacement = terminal.physics.clone()
    replacement.position[0, 1] += 1
    replaced = terminal.replace_physics(replacement)

    assert torch.equal(
        replaced.embodiment.previous_actions,
        terminal.embodiment.previous_actions,
    )
    assert torch.equal(
        replaced.embodiment.measured_acceleration,
        terminal.embodiment.measured_acceleration,
    )
    assert torch.equal(replaced.embodiment.extracted, terminal.embodiment.extracted)
    assert torch.equal(replaced.embodiment.failed, terminal.embodiment.failed)
    assert torch.equal(replaced.objectives.collected, terminal.objectives.collected)


def test_physics_replacement_cannot_resurrect_or_move_previously_dead_agent():
    state = _initialize()
    terminal = state.mark_failed(
        torch.tensor(
            [[[True], [False]], [[False], [False]]],
            dtype=torch.bool,
        )
    )

    resurrected = terminal.physics.clone()
    resurrected.alive[0, 0, 0] = True
    with pytest.raises(ValueError, match="reactivate dead"):
        terminal.replace_physics(resurrected)

    moved_dead = terminal.physics.clone()
    moved_dead.position[0, 0, 0] += 1
    with pytest.raises(ValueError, match="frozen physical state"):
        terminal.replace_physics(moved_dead)


def test_terminal_annotations_do_not_infer_or_change_other_agents():
    state = _initialize(energy=0.0)
    failed = state.mark_failed(
        torch.tensor(
            [[[False], [True]], [[False], [False]]],
            dtype=torch.bool,
        )
    )

    assert bool(failed.physics.alive[0, 0, 0].item())
    assert not bool(failed.embodiment.failed[0, 0, 0].item())
    assert not bool(failed.physics.alive[0, 1, 0].item())
    assert bool(failed.embodiment.failed[0, 1, 0].item())
    assert not bool(failed.embodiment.extracted.any().item())


def test_measurement_recording_requires_explicit_finite_difference_input():
    state = _initialize()
    actions = torch.full((2, 2, 2), -0.5)
    acceleration = torch.full((2, 2, 2), 1.25)

    updated = state.record_transition_measurements(
        applied_actions=actions,
        measured_acceleration=acceleration,
    )

    assert torch.equal(updated.embodiment.previous_actions, actions)
    assert torch.equal(updated.embodiment.measured_acceleration, acceleration)
    assert not torch.equal(
        updated.embodiment.measured_acceleration,
        state.embodiment.measured_acceleration,
    )

    with pytest.raises(ValueError, match="finite"):
        state.record_transition_measurements(
            applied_actions=actions,
            measured_acceleration=torch.full_like(acceleration, float("nan")),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_reset_clone_and_collection_storage_are_independent():
    state = _initialize(device="cuda")
    clone = state.clone()

    updated, newly = clone.mark_collected(
        torch.tensor([0, 0, 3], dtype=torch.int64, device="cuda"),
        torch.tensor([1, 1, 0], dtype=torch.int64, device="cuda"),
    )

    assert torch.equal(newly.cpu(), torch.tensor([True, False, True]))
    assert not bool(state.objectives.collected.any().item())
    assert bool(updated.objectives.collected[0, 0, 1].item())
    assert bool(updated.objectives.collected[1, 1, 0].item())

    reset = AgentEpisodeState.reset(
        state.generated_world,
        torch.tensor([[2, 1], [0, 2]], dtype=torch.int64, device="cuda"),
        embodiment_config=EmbodimentConfig(agent_radius_m=0.5),
        objective_config=ObjectiveConfig(core_count=2),
        physics_config=PhysicsConfig(),
    )
    assert reset.generated_world is state.generated_world
    assert not bool(reset.objectives.collected.any().item())
    assert bool(reset.physics.alive.all().item())


def test_spawn_assignment_rejects_negative_and_out_of_range_indices():
    layout = _layout()

    for indices in (
        torch.tensor([-1], dtype=torch.int64),
        torch.tensor([3], dtype=torch.int64),
    ):
        with pytest.raises(IndexError, match="spawn index out of range"):
            AgentEpisodeState.initialize(
                layout,
                indices,
                embodiment_config=EmbodimentConfig(agent_radius_m=0.5),
                objective_config=ObjectiveConfig(core_count=2),
                physics_config=PhysicsConfig(),
            )


from lawless.world.config import EnvironmentConfig
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


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
        pytest.param(
            "cuda",
            torch.float64,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ],
)
def test_real_generated_world_initialization_preserves_assignment_shape_and_layout(
    device, dtype
):
    layout = _generate_real_layout(device=device, dtype=dtype)
    embodiment_config, objective_config, physics_config = _real_episode_config()

    assignments = torch.tensor(
        [[0, 2], [1, 0]],
        dtype=torch.int64,
        device=device,
    )

    # Snapshot the shared layout's per-layout tensors before composition.
    spawn_snapshot = layout.spawn_positions.clone()
    core_position_snapshot = layout.core_positions.clone()
    core_radius_snapshot = layout.core_radii.clone()
    extraction_snapshot = layout.extraction_center.clone()

    state = AgentEpisodeState.initialize(
        layout,
        assignments,
        embodiment_config=embodiment_config,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    expected_positions = layout.spawn_positions.index_select(
        0, assignments.reshape(-1)
    ).reshape(2, 2, 2)

    assert state.generated_world is layout
    assert state.batch_shape == (2, 2)
    assert torch.equal(state.spawn_indices, assignments)
    assert state.physics.position.dtype == dtype
    assert state.physics.position.device == torch.device(device)
    assert torch.equal(state.physics.position, expected_positions)
    assert state.objectives.collected.shape == (2, 2, 2)
    assert torch.equal(state.physics.energy, torch.full(
        (2, 2, 1),
        80.0,
        dtype=dtype,
        device=device,
    ))

    # Initialization must not modify or replace shared layout geometry.
    assert torch.equal(layout.spawn_positions, spawn_snapshot)
    assert torch.equal(layout.core_positions, core_position_snapshot)
    assert torch.equal(layout.core_radii, core_radius_snapshot)
    assert torch.equal(layout.extraction_center, extraction_snapshot)


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
        pytest.param(
            "cuda",
            torch.float64,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA unavailable",
            ),
        ),
    ],
)
def test_real_generated_world_clone_collection_and_reset_are_isolated(
    device, dtype
):
    layout = _generate_real_layout(device=device, dtype=dtype)
    embodiment_config, objective_config, physics_config = _real_episode_config()

    assignments = torch.tensor(
        [[0, 1], [2, 0]],
        dtype=torch.int64,
        device=device,
    )
    state = AgentEpisodeState.initialize(
        layout,
        assignments,
        embodiment_config=embodiment_config,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    layout_spawns_before = layout.spawn_positions.clone()
    layout_cores_before = layout.core_positions.clone()

    clone = state.clone()
    assert clone.generated_world is layout
    assert clone.spawn_indices.data_ptr() != state.spawn_indices.data_ptr()
    assert clone.physics.position.data_ptr() != state.physics.position.data_ptr()
    assert (
        clone.embodiment.previous_actions.data_ptr()
        != state.embodiment.previous_actions.data_ptr()
    )
    assert (
        clone.embodiment.measured_acceleration.data_ptr()
        != state.embodiment.measured_acceleration.data_ptr()
    )
    assert (
        clone.objectives.collected.data_ptr()
        != state.objectives.collected.data_ptr()
    )

    clone.physics.position[0, 0, 0] += 4.0
    clone.spawn_indices[0, 0] = 2
    clone.embodiment.previous_actions[0, 0, 0] = 0.5
    clone.embodiment.measured_acceleration[0, 0, 0] = 3.0
    clone.objectives.collected[0, 0, 1] = True

    assert not torch.equal(clone.physics.position, state.physics.position)
    assert state.spawn_indices[0, 0].item() == 0
    assert state.embodiment.previous_actions[0, 0, 0].item() == 0.0
    assert state.embodiment.measured_acceleration[0, 0, 0].item() == 0.0
    assert not bool(state.objectives.collected.any().item())

    # Also exercise the vectorized objective update through the composition.
    collected, newly = state.mark_collected(
        torch.tensor([0, 0, 3], dtype=torch.int64, device=device),
        torch.tensor([1, 1, 0], dtype=torch.int64, device=device),
    )
    assert torch.equal(
        newly.cpu(),
        torch.tensor([True, False, True], dtype=torch.bool),
    )
    assert bool(collected.objectives.collected[0, 0, 1].item())
    assert bool(collected.objectives.collected[1, 1, 0].item())
    assert not bool(state.objectives.collected.any().item())

    reset_assignments = torch.tensor(
        [[2, 1], [0, 2]],
        dtype=torch.int64,
        device=device,
    )
    reset = AgentEpisodeState.reset(
        layout,
        reset_assignments,
        embodiment_config=embodiment_config,
        objective_config=objective_config,
        physics_config=PhysicsConfig(initial_energy=60.0),
    )

    expected_reset_positions = layout.spawn_positions.index_select(
        0, reset_assignments.reshape(-1)
    ).reshape(2, 2, 2)

    assert reset.generated_world is layout
    assert torch.equal(reset.spawn_indices, reset_assignments)
    assert torch.equal(reset.physics.position, expected_reset_positions)
    assert torch.equal(
        reset.physics.energy,
        torch.full((2, 2, 1), 60.0, dtype=dtype, device=device),
    )
    assert bool(reset.physics.alive.all().item())
    assert not bool(reset.embodiment.extracted.any().item())
    assert not bool(reset.embodiment.failed.any().item())
    assert not bool(reset.objectives.collected.any().item())
    assert not bool(reset.embodiment.previous_actions.any().item())
    assert not bool(reset.embodiment.measured_acceleration.any().item())

    # Reset and clone operations do not modify the shared layout.
    assert torch.equal(layout.spawn_positions, layout_spawns_before)
    assert torch.equal(layout.core_positions, layout_cores_before)


def test_real_generated_world_rejects_nonintegral_core_count_before_conversion():
    layout = _generate_real_layout()
    embodiment_config = EmbodimentConfig(agent_radius_m=0.5)
    invalid_objectives = ObjectiveConfig(core_count=2.9)

    with pytest.raises(ValueError, match="integer"):
        AgentEpisodeState.initialize(
            layout,
            torch.tensor([0], dtype=torch.int64),
            embodiment_config=embodiment_config,
            objective_config=invalid_objectives,
            physics_config=PhysicsConfig(),
        )


def test_real_generated_world_rejects_string_radius_before_conversion():
    layout = _generate_real_layout()
    invalid_embodiment = EmbodimentConfig(agent_radius_m="0.5")

    with pytest.raises(ValueError, match="numeric"):
        AgentEpisodeState.initialize(
            layout,
            torch.tensor([0], dtype=torch.int64),
            embodiment_config=invalid_embodiment,
            objective_config=ObjectiveConfig(core_count=2),
            physics_config=PhysicsConfig(),
        )


def test_direct_episode_construction_rejects_physics_dtype_mismatch():
    layout = _generate_real_layout(dtype=torch.float32)
    embodiment_config, objective_config, physics_config = _real_episode_config()
    state = AgentEpisodeState.initialize(
        layout,
        torch.tensor([0, 1], dtype=torch.int64),
        embodiment_config=embodiment_config,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    double_physics = AgentState(
        position=state.physics.position.double(),
        velocity=state.physics.velocity.double(),
        angle=state.physics.angle.double(),
        angular_velocity=state.physics.angular_velocity.double(),
        energy=state.physics.energy.double(),
        alive=state.physics.alive.clone(),
    )
    double_embodiment = AgentEmbodimentState.initialize(double_physics)

    with pytest.raises(TypeError, match="physics dtype must match"):
        AgentEpisodeState(
            generated_world=layout,
            spawn_indices=state.spawn_indices,
            embodiment=double_embodiment,
            objectives=state.objectives,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_direct_episode_construction_rejects_physics_device_mismatch():
    layout = _generate_real_layout(device="cpu", dtype=torch.float32)
    embodiment_config, objective_config, physics_config = _real_episode_config()
    state = AgentEpisodeState.initialize(
        layout,
        torch.tensor([0, 1], dtype=torch.int64),
        embodiment_config=embodiment_config,
        objective_config=objective_config,
        physics_config=physics_config,
    )

    cuda_physics = AgentState(
        position=state.physics.position.cuda(),
        velocity=state.physics.velocity.cuda(),
        angle=state.physics.angle.cuda(),
        angular_velocity=state.physics.angular_velocity.cuda(),
        energy=state.physics.energy.cuda(),
        alive=state.physics.alive.cuda(),
    )
    cuda_embodiment = AgentEmbodimentState.initialize(cuda_physics)
    cuda_spawn_indices = state.spawn_indices.cuda()
    cuda_objectives = AgentCoreCollectionState.initialize(
        state.batch_shape,
        state.objectives.core_count,
        device="cuda",
    )

    with pytest.raises(ValueError, match="physics device must match"):
        AgentEpisodeState(
            generated_world=layout,
            spawn_indices=cuda_spawn_indices,
            embodiment=cuda_embodiment,
            objectives=cuda_objectives,
        )
