import pytest
import torch

from lawless.physics.state import AgentState
from lawless.world.collision import (
    _tolerance,
    resolve_static_obstacles,
)
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState


def make_world(
    centers=None,
    radii=None,
    *,
    dtype=torch.float32,
    width=100.0,
    height=100.0,
):
    if centers is None:
        centers = torch.empty((0, 2), dtype=dtype)
        radii = torch.empty((0,), dtype=dtype)

    obstacles = CircleObstacles(
        centers=torch.tensor(centers, dtype=dtype).reshape(-1, 2),
        radii=torch.tensor(radii, dtype=dtype).reshape(-1),
    )
    return WorldState(width_m=width, height_m=height, obstacles=obstacles)


def make_physics(
    positions,
    velocities=None,
    *,
    alive=None,
    dtype=torch.float32,
    device="cpu",
):
    positions = torch.tensor(positions, dtype=dtype, device=device)
    if velocities is None:
        velocities = torch.zeros_like(positions)
    else:
        velocities = torch.tensor(velocities, dtype=dtype, device=device)

    batch_shape = positions.shape[:-1]
    if alive is None:
        alive = torch.ones(
            batch_shape + (1,), dtype=torch.bool, device=device
        )
    else:
        alive = torch.tensor(
            alive, dtype=torch.bool, device=device
        ).reshape(batch_shape + (1,))

    return AgentState(
        position=positions,
        velocity=velocities,
        angle=torch.zeros(batch_shape + (1,), dtype=dtype, device=device),
        angular_velocity=torch.zeros(
            batch_shape + (1,), dtype=dtype, device=device
        ),
        energy=torch.full(
            batch_shape + (1,), 100.0, dtype=dtype, device=device
        ),
        alive=alive,
    )


def test_single_obstacle_depenetration_and_inward_velocity_removal():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[53.0, 50.0]], [[-2.0, 3.0]])

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert torch.allclose(result.physics.position, torch.tensor([[56.0, 50.0]]))
    assert torch.allclose(result.physics.velocity, torch.tensor([[0.0, 3.0]]))
    assert result.contact.tolist() == [[True]]
    assert result.unresolved_contact.tolist() == [[False]]
    assert result.boundary_escape.tolist() == [[False]]


def test_outward_velocity_is_preserved():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[53.0, 50.0]], [[2.0, 3.0]])

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert torch.allclose(result.physics.position, torch.tensor([[56.0, 50.0]]))
    assert torch.allclose(result.physics.velocity, torch.tensor([[2.0, 3.0]]))
    assert result.contact.tolist() == [[True]]


def test_exact_tangency_removes_inward_velocity_without_position_correction():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[56.0, 50.0]], [[-2.0, 3.0]])

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert torch.equal(result.physics.position, physics.position)
    assert torch.allclose(result.physics.velocity, torch.tensor([[0.0, 3.0]]))
    assert result.contact.tolist() == [[True]]
    assert result.unresolved_contact.tolist() == [[False]]


def test_tangent_outward_velocity_is_preserved():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[56.0, 50.0]], [[2.0, 3.0]])

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert torch.equal(result.physics.position, physics.position)
    assert torch.equal(result.physics.velocity, physics.velocity)
    assert result.contact.tolist() == [[True]]


def test_tangent_velocity_parallel_to_normal_is_not_modified_when_tangential():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[56.0, 50.0]], [[0.0, 3.0]])

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert torch.equal(result.physics.position, physics.position)
    assert torch.equal(result.physics.velocity, physics.velocity)
    assert result.contact.tolist() == [[True]]


def test_clear_agent_reports_no_contact():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[10.0, 10.0]], [[1.0, 2.0]])

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert result.contact.tolist() == [[False]]
    assert result.unresolved_contact.tolist() == [[False]]
    assert result.boundary_escape.tolist() == [[False]]


def test_coincident_centers_use_deterministic_fallback():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[50.0, 50.0]], [[-1.0, 2.0]])

    first = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )
    second = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert torch.equal(first.physics.position, second.physics.position)
    assert torch.equal(first.physics.velocity, second.physics.velocity)
    assert torch.allclose(first.physics.position, torch.tensor([[56.0, 50.0]]))
    assert torch.allclose(first.physics.velocity, torch.tensor([[0.0, 2.0]]))
    assert first.contact.tolist() == [[True]]


def test_dead_agents_are_frozen_and_excluded_from_contact():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics(
        [[53.0, 50.0], [53.0, 50.0]],
        [[-2.0, 1.0], [-2.0, 1.0]],
        alive=[True, False],
    )

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert torch.allclose(result.physics.position[0], torch.tensor([56.0, 50.0]))
    assert torch.equal(result.physics.position[1], physics.position[1])
    assert torch.equal(result.physics.velocity[1], physics.velocity[1])
    assert result.contact.tolist() == [[True], [False]]
    assert result.unresolved_contact.tolist() == [[False], [False]]


def test_boundary_escape_is_reported_and_not_corrected():
    world = make_world([[5.0, 50.0]], [4.0])
    physics = make_physics([[-1.0, 50.0]], [[-2.0, 0.0]])

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert result.boundary_escape.tolist() == [[True]]
    assert result.physics.position[0, 0] < 0.0


def test_verified_obstacle_correction_introduces_strict_boundary_escape():
    # Fully contained obstacle. This fixture is locally verified.
    world = make_world([[4.0, 50.0]], [4.0])
    physics = make_physics([[3.0, 50.0]], [[0.0, 0.0]])

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert torch.equal(result.physics.position, torch.tensor([[-1.0, 50.0]]))
    assert result.boundary_escape.tolist() == [[True]]
    assert result.contact.tolist() == [[True]]


def test_input_boundary_escape_is_preserved():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[-1.0, 50.0]], [[0.0, 1.0]])

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert result.boundary_escape.tolist() == [[True]]
    assert result.physics.position[0, 0] < 0.0


def test_zero_obstacles_leave_physics_unchanged():
    world = make_world()
    physics = make_physics(
        [[10.0, 10.0], [50.0, 50.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    )

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert torch.equal(result.physics.position, physics.position)
    assert torch.equal(result.physics.velocity, physics.velocity)
    assert not result.contact.any()
    assert not result.unresolved_contact.any()
    assert not result.boundary_escape.any()


def test_batched_leading_dimensions_and_report_shapes_are_preserved():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics(
        [
            [[53.0, 50.0], [10.0, 10.0]],
            [[53.0, 50.0], [20.0, 20.0]],
        ]
    )

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert result.physics.position.shape == (2, 2, 2)
    assert result.contact.shape == (2, 2, 1)
    assert result.unresolved_contact.shape == (2, 2, 1)
    assert result.boundary_escape.shape == (2, 2, 1)
    assert result.contact.dtype == torch.bool
    assert result.unresolved_contact.dtype == torch.bool
    assert result.boundary_escape.dtype == torch.bool


def test_far_away_batch_outlier_does_not_change_local_resolution():
    world = make_world([[50.0, 50.0]], [5.0])
    near_only = make_physics([[53.0, 50.0]])
    near_plus_outlier = make_physics(
        [[53.0, 50.0], [1.0e7, 1.0e7]]
    )

    alone = resolve_static_obstacles(
        near_only, world, agent_radius_m=1.0
    )
    batched = resolve_static_obstacles(
        near_plus_outlier, world, agent_radius_m=1.0
    )

    assert torch.equal(alone.physics.position[0], batched.physics.position[0])
    assert torch.equal(alone.physics.velocity[0], batched.physics.velocity[0])
    assert torch.equal(alone.contact[0], batched.contact[0])
    assert torch.equal(
        alone.unresolved_contact[0], batched.unresolved_contact[0]
    )


def test_float64_resolution():
    world = make_world([[50.0, 50.0]], [5.0], dtype=torch.float64)
    physics = make_physics(
        [[53.0, 50.0]], [[-2.0, 3.0]], dtype=torch.float64
    )

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert result.physics.position.dtype == torch.float64
    assert torch.allclose(
        result.physics.position,
        torch.tensor([[56.0, 50.0]], dtype=torch.float64),
    )
    assert torch.allclose(
        result.physics.velocity,
        torch.tensor([[0.0, 3.0]], dtype=torch.float64),
    )


def test_converged_penetration_reports_contact_not_unresolved():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[53.0, 50.0]], [[-2.0, 3.0]])

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert result.contact.tolist() == [[True]]
    assert result.unresolved_contact.shape == (1, 1)
    assert result.unresolved_contact.tolist() == [[False]]


def test_verified_two_obstacle_fixture_remains_unresolved_after_one_pass():
    world = make_world(
        [[45.0, 50.0], [55.0, 50.0]],
        [4.0, 4.0],
    )
    physics = make_physics([[50.0, 50.0]], [[0.0, 0.0]])

    result = resolve_static_obstacles(
        physics,
        world,
        agent_radius_m=2.0,
        max_passes=1,
    )

    assert result.unresolved_contact.shape == (1, 1)
    assert result.unresolved_contact.dtype == torch.bool
    assert result.unresolved_contact.tolist() == [[True]]
    assert result.contact.tolist() == [[True]]


def test_verified_two_obstacle_fixture_remains_unresolved_after_eight_passes():
    world = make_world(
        [[45.0, 50.0], [55.0, 50.0]],
        [4.0, 4.0],
    )
    physics = make_physics([[50.0, 50.0]], [[0.0, 0.0]])

    result = resolve_static_obstacles(
        physics,
        world,
        agent_radius_m=2.0,
        max_passes=8,
    )

    assert result.unresolved_contact.shape == (1, 1)
    assert result.unresolved_contact.tolist() == [[True]]
    assert result.contact.tolist() == [[True]]


def test_multi_contact_resolution_is_deterministic():
    world = make_world(
        [[45.0, 50.0], [55.0, 50.0]],
        [4.0, 4.0],
    )
    physics = make_physics([[50.0, 50.0]], [[-2.0, 3.0]])

    first = resolve_static_obstacles(
        physics, world, agent_radius_m=2.0
    )
    second = resolve_static_obstacles(
        physics, world, agent_radius_m=2.0
    )

    assert torch.equal(first.physics.position, second.physics.position)
    assert torch.equal(first.physics.velocity, second.physics.velocity)
    assert torch.equal(first.contact, second.contact)
    assert torch.equal(first.unresolved_contact, second.unresolved_contact)
    assert torch.equal(first.boundary_escape, second.boundary_escape)


def test_zero_restitution_contact_does_not_add_kinetic_energy():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[56.0, 50.0]], [[-2.0, 3.0]])

    before = physics.velocity.square().sum(dim=-1)
    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )
    after = result.physics.velocity.square().sum(dim=-1)

    assert torch.all(after <= before)


def test_solver_does_not_mutate_input_physics_or_layout():
    world = make_world([[50.0, 50.0]], [5.0])
    physics = make_physics([[53.0, 50.0]], [[-2.0, 3.0]])

    original_position = physics.position.clone()
    original_velocity = physics.velocity.clone()
    original_angle = physics.angle.clone()
    original_angular_velocity = physics.angular_velocity.clone()
    original_energy = physics.energy.clone()
    original_alive = physics.alive.clone()
    original_centers = world.obstacles.centers.clone()
    original_radii = world.obstacles.radii.clone()

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    assert torch.equal(physics.position, original_position)
    assert torch.equal(physics.velocity, original_velocity)
    assert torch.equal(physics.angle, original_angle)
    assert torch.equal(physics.angular_velocity, original_angular_velocity)
    assert torch.equal(physics.energy, original_energy)
    assert torch.equal(physics.alive, original_alive)
    assert torch.equal(world.obstacles.centers, original_centers)
    assert torch.equal(world.obstacles.radii, original_radii)

    assert result.physics.position.data_ptr() != physics.position.data_ptr()
    assert result.physics.velocity.data_ptr() != physics.velocity.data_ptr()


@pytest.mark.parametrize(
    "radius",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        True,
    ],
)
def test_invalid_agent_radius_rejected(radius):
    world = make_world()
    physics = make_physics([[10.0, 10.0]])

    with pytest.raises((TypeError, ValueError)):
        resolve_static_obstacles(
            physics, world, agent_radius_m=radius
        )


def test_float32_agent_radius_underflow_rejected():
    world = make_world()
    physics = make_physics([[10.0, 10.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="underflows"):
        resolve_static_obstacles(
            physics,
            world,
            agent_radius_m=1e-100,
        )


@pytest.mark.parametrize("max_passes", [0, -1])
def test_invalid_pass_limit_rejected(max_passes):
    world = make_world()
    physics = make_physics([[10.0, 10.0]])

    with pytest.raises(ValueError):
        resolve_static_obstacles(
            physics, world, agent_radius_m=1.0, max_passes=max_passes
        )


def test_noninteger_pass_limit_rejected():
    world = make_world()
    physics = make_physics([[10.0, 10.0]])

    with pytest.raises(TypeError):
        resolve_static_obstacles(
            physics, world, agent_radius_m=1.0, max_passes=1.5
        )


def test_malformed_agent_field_type_is_guarded_before_shape_validation():
    physics = make_physics([[10.0, 10.0]])
    physics.velocity = None
    world = make_world()

    with pytest.raises(TypeError, match="physics.velocity"):
        resolve_static_obstacles(
            physics, world, agent_radius_m=1.0
        )


def test_invalid_alive_dtype_remains_rejected_by_agent_state_validation():
    physics = make_physics([[10.0, 10.0]])
    physics.alive = physics.alive.to(torch.float32)
    world = make_world()

    with pytest.raises(TypeError, match="alive"):
        resolve_static_obstacles(
            physics, world, agent_radius_m=1.0
        )


def test_tolerance_scale_floor_at_unit_scale():
    positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    layout_scale = torch.tensor(0.0, dtype=torch.float32)

    tolerance = _tolerance(positions, 0.25, layout_scale)

    assert torch.allclose(
        tolerance,
        torch.tensor([[8.0 * torch.finfo(torch.float32).eps]]),
    )


def test_tolerance_scales_per_agent_without_batch_max():
    positions = torch.tensor(
        [[100.0, 0.0], [1.0e7, 1.0e7]],
        dtype=torch.float32,
    )
    layout_scale = torch.tensor(100.0, dtype=torch.float32)

    tolerances = _tolerance(positions, 1.0, layout_scale)
    alone = _tolerance(positions[:1], 1.0, layout_scale)

    assert torch.equal(tolerances[:1], alone)
    assert tolerances[1, 0] > tolerances[0, 0]


def test_float32_100m_scale_threshold_representation():
    positions = torch.tensor([[100.0, 0.0]], dtype=torch.float32)
    layout_scale = torch.tensor(100.0, dtype=torch.float32)

    tolerance = _tolerance(positions, 1.0, layout_scale)
    epsilon = torch.finfo(torch.float32).eps

    assert torch.allclose(
        tolerance, torch.tensor([[8.0 * epsilon * 100.0]])
    )

    assert tolerance.item() > (
        torch.nextafter(
            torch.tensor(100.0), torch.tensor(float("inf"))
        ).item() - 100.0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_exact_resolution_and_concrete_device(dtype):
    device = torch.device("cuda:0")
    world = make_world(
        [[50.0, 50.0]], [5.0], dtype=dtype
    )
    # Move the real layout to CUDA, preserving dtype.
    world = WorldState(
        width_m=world.width_m,
        height_m=world.height_m,
        obstacles=CircleObstacles(
            centers=world.obstacles.centers.to(device=device),
            radii=world.obstacles.radii.to(device=device),
        ),
    )
    physics = make_physics(
        [[53.0, 50.0]], [[-2.0, 3.0]], dtype=dtype, device=device
    )

    result = resolve_static_obstacles(
        physics, world, agent_radius_m=1.0
    )

    expected_position = torch.tensor(
        [[56.0, 50.0]], dtype=dtype, device=device
    )
    expected_velocity = torch.tensor(
        [[0.0, 3.0]], dtype=dtype, device=device
    )

    assert result.physics.position.device == expected_position.device
    assert result.physics.velocity.device == expected_velocity.device
    assert torch.equal(result.physics.position, expected_position)
    assert torch.equal(result.physics.velocity, expected_velocity)
    assert result.contact.tolist() == [[True]]
    assert result.unresolved_contact.tolist() == [[False]]
    assert result.boundary_escape.tolist() == [[False]]

from lawless.physics.constants import PhysicsConfig
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


def _real_episode_state():
    layout = _generate_real_layout()
    embodiment_config, objective_config, physics_config = (
        _real_episode_config()
    )

    spawn_indices = torch.tensor(
        [[0, 1], [1, 2]], dtype=torch.int64
    )

    return AgentEpisodeState.initialize(
        layout,
        spawn_indices,
        embodiment_config=embodiment_config,
        objective_config=objective_config,
        physics_config=physics_config,
    )


def test_real_episode_physics_replacement_preserves_objective_history_and_lifecycle():
    state = _real_episode_state()

    collected, newly = state.mark_collected(
        torch.tensor([0, 1, 0], dtype=torch.int64),
        torch.tensor([1, 1, 1], dtype=torch.int64),
    )
    assert newly.tolist() == [True, True, False]

    measured = collected.record_transition_measurements(
        applied_actions=torch.full((2, 2, 2), 0.25),
        measured_acceleration=torch.full((2, 2, 2), 2.5),
    )

    terminal = measured.mark_failed(
        torch.tensor(
            [[[False], [True]], [[False], [False]]],
            dtype=torch.bool,
        )
    )

    # No obstacle correction occurs in this real layout. This regression
    # protects the composition API during collision-resolution replacement.
    result = resolve_static_obstacles(
        terminal.physics,
        terminal.generated_world.world,
        agent_radius_m=0.5,
    )
    replaced = terminal.replace_physics(result.physics)

    assert replaced.generated_world is terminal.generated_world
    assert torch.equal(replaced.spawn_indices, terminal.spawn_indices)

    assert torch.equal(
        replaced.objectives.collected,
        terminal.objectives.collected,
    )
    assert torch.equal(
        replaced.embodiment.previous_actions,
        terminal.embodiment.previous_actions,
    )
    assert torch.equal(
        replaced.embodiment.measured_acceleration,
        terminal.embodiment.measured_acceleration,
    )
    assert torch.equal(
        replaced.embodiment.extracted,
        terminal.embodiment.extracted,
    )
    assert torch.equal(
        replaced.embodiment.failed,
        terminal.embodiment.failed,
    )

    # The composition clones owned per-agent state.
    assert (
        replaced.objectives.collected.data_ptr()
        != terminal.objectives.collected.data_ptr()
    )


def test_real_episode_dead_agent_cannot_be_resurrected_by_collision_replacement():
    state = _real_episode_state()
    failed = state.mark_failed(
        torch.tensor(
            [[[True], [False]], [[False], [False]]],
            dtype=torch.bool,
        )
    )

    result = resolve_static_obstacles(
        failed.physics,
        failed.generated_world.world,
        agent_radius_m=0.5,
    )

    attempted_resurrection = result.physics.clone()
    attempted_resurrection.alive[0, 0, 0] = True

    with pytest.raises(ValueError, match="reactivate dead"):
        failed.replace_physics(attempted_resurrection)


def test_real_episode_dead_agent_physical_state_remains_frozen():
    state = _real_episode_state()

    failed = state.mark_failed(
        torch.tensor(
            [[[True], [False]], [[False], [False]]],
            dtype=torch.bool,
        )
    )

    before = failed.physics.clone()
    result = resolve_static_obstacles(
        failed.physics,
        failed.generated_world.world,
        agent_radius_m=0.5,
    )
    replaced = failed.replace_physics(result.physics)

    assert torch.equal(
        replaced.physics.position[0, 0],
        before.position[0, 0],
    )
    assert torch.equal(
        replaced.physics.velocity[0, 0],
        before.velocity[0, 0],
    )
    assert not replaced.physics.alive[0, 0, 0]


def test_episode_radius_must_match_generated_world_at_composition_boundary():
    layout = _generate_real_layout()
    spawn_indices = torch.tensor([[0, 1]], dtype=torch.int64)
    _, objective_config, physics_config = _real_episode_config()

    mismatched_embodiment_config = EmbodimentConfig(
        agent_radius_m=0.75
    )

    with pytest.raises(ValueError, match="does not match"):
        AgentEpisodeState.initialize(
            layout,
            spawn_indices,
            embodiment_config=mismatched_embodiment_config,
            objective_config=objective_config,
            physics_config=physics_config,
        )


@pytest.mark.parametrize("max_passes", [0, 9, 100])
def test_max_passes_rejects_values_outside_selected_limit(max_passes):
    physics = make_physics([[53.0, 50.0]])
    world = make_world(
        centers=[[50.0, 50.0]],
        radii=[5.0],
    )

    with pytest.raises(ValueError, match="max_passes"):
        resolve_static_obstacles(
            physics,
            world,
            agent_radius_m=1.0,
            max_passes=max_passes,
        )

@pytest.mark.parametrize("max_passes", [True, 1.5])
def test_max_passes_rejects_non_integer_values(max_passes):
    physics = make_physics([[53.0, 50.0]])
    world = make_world(
        centers=[[50.0, 50.0]],
        radii=[5.0],
    )

    with pytest.raises(TypeError, match="max_passes"):
        resolve_static_obstacles(
            physics,
            world,
            agent_radius_m=1.0,
            max_passes=max_passes,
        )

@pytest.mark.parametrize("max_passes", [1, 8])
def test_max_passes_accepts_selected_range_endpoints(max_passes):
    physics = make_physics([[53.0, 50.0]])
    world = make_world(
        centers=[[50.0, 50.0]],
        radii=[5.0],
    )

    result = resolve_static_obstacles(
        physics,
        world,
        agent_radius_m=1.0,
        max_passes=max_passes,
    )

    assert result.physics.position.shape == physics.position.shape


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
def test_collision_resolution_rejects_nonfinite_physics_fields(field):
    physics = make_physics([[50.0, 50.0]])
    corrupted = physics.clone()

    getattr(corrupted, field)[0, 0] = float("nan")

    world = make_world()

    with pytest.raises(ValueError, match=f"physics\\.{field}"):
        resolve_static_obstacles(
            corrupted,
            world,
            agent_radius_m=1.0,
        )
