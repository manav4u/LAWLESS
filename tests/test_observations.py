
"""P1.6 observation assembly contract and numerical regressions."""

from __future__ import annotations

from dataclasses import replace
import math

import pytest
import torch

from lawless.physics.constants import PhysicsConfig
from lawless.physics.state import AgentState
from lawless.sensors import assemble_observations, sense_lidar
from lawless.sensors.lidar import SensorConfig
from lawless.world.config import (
    EmbodimentConfig,
    ObjectiveConfig,
)
from lawless.world.episode_state import AgentEpisodeState
from lawless.world.generated_world import GeneratedWorld
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState


DTYPES = (torch.float32, torch.float64)
DEVICES = ("cpu", "cuda")


def _make_episode(
    *,
    dtype=torch.float32,
    device="cpu",
    batch_shape=(1,),
    initial_energy=100.0,
    extraction_center=(80.0, 80.0),
    extraction_radius=3.0,
    core_centers=((40.0, 50.0), (44.0, 50.0)),
    core_radii=(0.75, 0.75),
):
    """Construct a valid controlled episode using public constructors."""
    device = torch.device(device)

    obstacles = CircleObstacles(
        centers=torch.empty((0, 2), dtype=dtype, device=device),
        radii=torch.empty((0,), dtype=dtype, device=device),
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
        ).reshape(-1),
        seed=12,
    )

    embodiment_config = EmbodimentConfig(agent_radius_m=0.5)
    objective_config = ObjectiveConfig(core_count=len(core_radii))
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
    return state


def _replace_physics(state, **changes):
    """Return an episode with a replaced, validated physical state."""
    physics = replace(state.embodiment.physics, **changes)
    return state.replace_physics(physics)


def _set_positions(state, positions):
    positions = torch.as_tensor(
        positions,
        dtype=state.embodiment.physics.position.dtype,
        device=state.embodiment.physics.position.device,
    )
    return _replace_physics(state, position=positions)


def _set_angles(state, angles):
    angles = torch.as_tensor(
        angles,
        dtype=state.embodiment.physics.angle.dtype,
        device=state.embodiment.physics.angle.device,
    )
    return _replace_physics(state, angle=angles)


def _set_velocity(state, velocity):
    velocity = torch.as_tensor(
        velocity,
        dtype=state.embodiment.physics.velocity.dtype,
        device=state.embodiment.physics.velocity.device,
    )
    return _replace_physics(state, velocity=velocity)


def _set_angular_velocity(state, angular_velocity):
    angular_velocity = torch.as_tensor(
        angular_velocity,
        dtype=state.embodiment.physics.angular_velocity.dtype,
        device=state.embodiment.physics.angular_velocity.device,
    )
    return _replace_physics(state, angular_velocity=angular_velocity)


def _set_energy(state, energy):
    energy = torch.as_tensor(
        energy,
        dtype=state.embodiment.physics.energy.dtype,
        device=state.embodiment.physics.energy.device,
    )
    return _replace_physics(state, energy=energy)


def _set_lifecycle(state, *, alive, extracted=None, failed=None):
    """Construct lifecycle annotations without inferring them from energy."""
    physics = replace(
        state.embodiment.physics,
        alive=torch.as_tensor(
            alive,
            dtype=torch.bool,
            device=state.embodiment.physics.position.device,
        ),
    )
    extracted_tensor = (
        state.embodiment.extracted.clone()
        if extracted is None
        else torch.as_tensor(
            extracted,
            dtype=torch.bool,
            device=physics.position.device,
        )
    )
    failed_tensor = (
        state.embodiment.failed.clone()
        if failed is None
        else torch.as_tensor(
            failed,
            dtype=torch.bool,
            device=physics.position.device,
        )
    )
    embodiment = replace(
        state.embodiment,
        physics=physics,
        extracted=extracted_tensor,
        failed=failed_tensor,
    )
    return replace(state, embodiment=embodiment)


def _assemble(state, lidar=None):
    if lidar is None:
        lidar = torch.full(
            tuple(state.batch_shape) + (8,),
            0.25,
            dtype=state.embodiment.physics.position.dtype,
            device=state.embodiment.physics.position.device,
        )
    return assemble_observations(state, lidar)


def _cuda_available():
    return torch.cuda.is_available()


@pytest.fixture(params=DTYPES)
def dtype(request):
    return request.param


@pytest.fixture(params=DEVICES)
def device(request):
    if request.param == "cuda" and not _cuda_available():
        pytest.skip("CUDA is unavailable")
    return torch.device(request.param)


@pytest.fixture
def episode_factory():
    return _make_episode


# ---------------------------------------------------------------------------
# Exact observation shape, order, and baseline channel values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device_name", DEVICES)
def test_observation_shape_dtype_device(dtype, device_name):
    if device_name == "cuda" and not _cuda_available():
        pytest.skip("CUDA is unavailable")

    state = _make_episode(dtype=dtype, device=device_name)
    observations = _assemble(state)

    assert observations.shape == (1, 23)
    assert observations.dtype == dtype
    assert observations.device == state.embodiment.physics.position.device
    assert torch.isfinite(observations).all()


def test_exact_23_channel_order_and_expected_baseline_values():
    state = _make_episode(dtype=torch.float64)
    lidar = torch.tensor(
        [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0]],
        dtype=torch.float64,
    )
    state.embodiment.previous_actions.copy_(
        torch.tensor([[0.25, -0.5]], dtype=torch.float64)
    )

    observations = assemble_observations(state, lidar)

    assert observations.shape == (1, 23)

    # Target is extraction center (80, 80), agent begins at (10, 10).
    assert torch.equal(
        observations[0, 0:2],
        torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    assert observations[0, 2].item() == pytest.approx(
        math.sqrt(70.0**2 + 70.0**2) / 100.0,
        abs=1e-12,
    )
    assert observations[0, 3].item() == pytest.approx(
        math.sqrt(0.5), abs=1e-12
    )
    assert observations[0, 4].item() == pytest.approx(
        math.sqrt(0.5), abs=1e-12
    )

    assert torch.equal(
        observations[0, 5:8],
        torch.zeros(3, dtype=torch.float64),
    )
    assert observations[0, 8].item() == pytest.approx(0.0)
    assert observations[0, 9].item() == pytest.approx(1.0)
    assert torch.equal(observations[0, 10:18], lidar[0])
    assert torch.equal(
        observations[0, 18:20],
        torch.tensor([0.25, -0.5], dtype=torch.float64),
    )
    assert torch.equal(
        observations[0, 20:22],
        torch.zeros(2, dtype=torch.float64),
    )
    assert observations[0, 22].item() == pytest.approx(1.0)


@pytest.mark.parametrize("batch_shape", [(5,), (2, 3), (3, 2)])
def test_nontrivial_batch_shapes(batch_shape):
    state = _make_episode(
        dtype=torch.float64,
        batch_shape=batch_shape,
    )
    lidar = torch.full(
        batch_shape + (8,),
        0.4,
        dtype=torch.float64,
    )

    observations = assemble_observations(state, lidar)

    assert observations.shape == batch_shape + (23,)
    assert observations.dtype == torch.float64
    assert torch.isfinite(observations).all()


# ---------------------------------------------------------------------------
# Extraction target remains independent of core collection
# ---------------------------------------------------------------------------

def test_extraction_center_is_target_not_nearest_uncollected_core():
    state = _make_episode(
        dtype=torch.float64,
        extraction_center=(80.0, 80.0),
        core_centers=((40.0, 50.0), (44.0, 50.0)),
    )
    before = _assemble(state)

    # Collect both cores for this agent. The observation target must not move.
    state.objectives.collected[0, 0] = True
    state.objectives.collected[0, 1] = True
    after = _assemble(state)

    assert torch.equal(before, after)
    assert before[0, 0].item() == pytest.approx(1.0)
    assert before[0, 1].item() == pytest.approx(1.0)


def test_extraction_target_is_used_when_no_cores_exist():
    state = _make_episode(
        dtype=torch.float64,
        extraction_center=(30.0, 10.0),
        core_centers=(),
        core_radii=(),
    )
    observations = _assemble(state)

    assert observations[0, 0].item() == pytest.approx(0.4)
    assert observations[0, 1].item() == pytest.approx(0.0)
    assert observations[0, 2].item() == pytest.approx(0.2)
    assert observations[0, 3].item() == pytest.approx(0.0)
    assert observations[0, 4].item() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fixed scales, clipping, and energy policy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_fixed_energy_denominator_independent_of_initial_energy(dtype):
    state = _make_episode(
        dtype=dtype,
        initial_energy=25.0,
    )
    observations = _assemble(state)

    assert observations[0, 22].item() == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("energy", "expected"),
    [
        (0.0, 0.0),
        (25.0, 0.25),
        (100.0, 1.0),
        (150.0, 1.0),
    ],
)
def test_energy_fraction_uses_fixed_scale_and_clips(energy, expected):
    state = _make_episode(dtype=torch.float64)
    state = _set_energy(state, [[energy]])

    observations = _assemble(state)
    assert observations[0, 22].item() == pytest.approx(expected)


def test_zero_energy_does_not_itself_make_agent_inactive():
    state = _make_episode(dtype=torch.float64, initial_energy=0.0)

    # make_state(energy=0) is allowed to retain alive=True until physics
    # performs a step. Observation masking follows physics.alive only.
    assert state.embodiment.physics.alive.item()
    observations = _assemble(state)

    assert observations[0, 9].item() == pytest.approx(1.0)
    assert observations[0, 4].item() > 0.0
    assert observations[0, 22].item() == 0.0


@pytest.mark.parametrize("dtype", DTYPES)
def test_velocity_angular_velocity_and_acceleration_fixed_scales(dtype):
    state = _make_episode(dtype=dtype)

    state = _set_velocity(state, [[6.0, -12.0]])
    state = _set_angular_velocity(state, [[2.0]])
    state.embodiment.measured_acceleration.copy_(
        torch.tensor([[8.0, -16.0]], dtype=dtype)
    )

    observations = _assemble(state)

    assert observations[0, 5].item() == pytest.approx(0.5)
    assert observations[0, 6].item() == pytest.approx(-1.0)
    assert observations[0, 7].item() == pytest.approx(0.5)
    assert observations[0, 20].item() == pytest.approx(0.5)
    assert observations[0, 21].item() == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Acceleration history and current-heading body-frame transformation
# ---------------------------------------------------------------------------

def test_reset_acceleration_is_zero_no_history_sentinel():
    state = _make_episode(dtype=torch.float64)

    assert torch.equal(
        state.embodiment.measured_acceleration,
        torch.zeros((1, 2), dtype=torch.float64),
    )

    observations = _assemble(state)
    assert torch.equal(
        observations[0, 20:22],
        torch.zeros(2, dtype=torch.float64),
    )


def test_completed_transition_measurement_is_observed_in_current_heading():
    state = _make_episode(dtype=torch.float64)
    state = _set_angles(state, [[math.pi / 2]])
    applied_actions = torch.tensor([[0.5, -0.25]], dtype=torch.float64)
    measured_acceleration = torch.tensor([[16.0, 0.0]], dtype=torch.float64)

    state = state.record_transition_measurements(
        applied_actions=applied_actions,
        measured_acceleration=measured_acceleration,
    )

    observations = _assemble(state)

    # At +90 degrees, world +X points to body -Y.
    assert observations[0, 20].item() == pytest.approx(0.0, abs=1e-12)
    assert observations[0, 21].item() == pytest.approx(-1.0, abs=1e-12)
    assert torch.equal(observations[0, 18:20], applied_actions[0])


def test_acceleration_uses_post_transition_heading_not_old_heading():
    state = _make_episode(dtype=torch.float64)
    state = _set_angles(state, [[0.0]])

    state = state.record_transition_measurements(
        applied_actions=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        measured_acceleration=torch.tensor([[16.0, 0.0]], dtype=torch.float64),
    )

    # The stored world-frame measurement is rotated using the current heading.
    state = _set_angles(state, [[math.pi]])

    observations = _assemble(state)
    assert observations[0, 20].item() == pytest.approx(-1.0, abs=1e-12)
    assert observations[0, 21].item() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Target displacement, bearing, and adjacent representable coordinates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_target_displacement_uses_forward_bearing_sentinel(dtype):
    state = _make_episode(dtype=dtype, extraction_center=(80.0, 80.0))
    state = _set_positions(state, [[80.0, 80.0]])
    observations = _assemble(state)

    assert observations[0, 2].item() == pytest.approx(0.0)
    assert observations[0, 3].item() == pytest.approx(0.0)
    assert observations[0, 4].item() == pytest.approx(1.0)


def test_adjacent_float64_target_positions_preserve_nonzero_displacement():
    state = _make_episode(
        dtype=torch.float64,
        extraction_center=(7.0, 7.0),
    )

    adjacent = torch.nextafter(
        torch.tensor(7.0, dtype=torch.float64),
        torch.tensor(float("inf"), dtype=torch.float64),
    )
    assert adjacent.item() != 7.0
    assert (adjacent - 7.0).item() == pytest.approx(
        8.881784197001252e-16,
        rel=0.0,
        abs=0.0,
    )

    # The target lies directly behind the agent by one representable step.
    state = _set_positions(state, [[adjacent.item(), 7.0]])
    observations = _assemble(state)

    assert observations[0, 2].item() > 0.0
    assert observations[0, 3].item() == pytest.approx(0.0, abs=1e-15)
    assert observations[0, 4].item() == pytest.approx(-1.0, abs=1e-15)
    assert observations[0, 0].item() < 0.0
    assert torch.isfinite(observations).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_bearing_cardinal_directions(dtype):
    state = _make_episode(dtype=dtype, extraction_center=(50.0, 50.0))

    cases = [
        ((40.0, 50.0), (0.0, 1.0)),   # target is body/world +X
        ((60.0, 50.0), (0.0, -1.0)),  # target is body/world -X
        ((50.0, 40.0), (1.0, 0.0)),   # target is body/world +Y
        ((50.0, 60.0), (-1.0, 0.0)),  # target is body/world -Y
    ]

    for position, expected in cases:
        positioned = _set_positions(state, [position])
        observations = _assemble(positioned)

        assert observations[0, 3].item() == pytest.approx(
            expected[0], abs=1e-6
        )
        assert observations[0, 4].item() == pytest.approx(
            expected[1], abs=1e-6
        )


@pytest.mark.parametrize("dtype", DTYPES)
def test_target_body_rotation_uses_negative_heading(dtype):
    state = _make_episode(
        dtype=dtype,
        extraction_center=(20.0, 10.0),
    )
    state = _set_angles(state, [[math.pi / 2]])

    observations = _assemble(state)

    # World +X becomes body -Y at a +90 degree heading.
    assert observations[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert observations[0, 1].item() == pytest.approx(-0.2, abs=1e-6)
    assert observations[0, 3].item() == pytest.approx(-1.0, abs=1e-6)
    assert observations[0, 4].item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Extreme finite values and numerical stability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_opposite_sign_near_maximum_target_coordinates_do_not_overflow(dtype):
    state = _make_episode(dtype=dtype)

    maximum = torch.finfo(dtype).max
    # Mutate the shared layout tensor only for this numerical stress case.
    state.generated_world.extraction_center.copy_(
        torch.tensor([maximum, maximum], dtype=dtype)
    )
    state = _set_positions(state, [[-maximum, -maximum]])

    observations = _assemble(state)

    assert torch.isfinite(observations).all()
    assert observations[0, 0].item() == pytest.approx(1.0)
    assert observations[0, 1].item() == pytest.approx(1.0)
    assert observations[0, 2].item() == pytest.approx(1.0)
    assert observations[0, 3].item() == pytest.approx(
        math.sqrt(0.5), abs=1e-6
    )
    assert observations[0, 4].item() == pytest.approx(
        math.sqrt(0.5), abs=1e-6
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_unequal_coordinate_magnitudes_preserve_subnormal_bearing(dtype):
    state = _make_episode(dtype=dtype)

    maximum = torch.finfo(dtype).max
    tiny = torch.nextafter(
        torch.tensor(0.0, dtype=dtype),
        torch.tensor(1.0, dtype=dtype),
    )

    state.generated_world.extraction_center.copy_(
        torch.tensor(
            [maximum, tiny.item()],
            dtype=dtype,
            device=state.embodiment.physics.position.device,
        )
    )
    state = _set_positions(
        state,
        torch.tensor(
            [[maximum, 0.0]],
            dtype=dtype,
            device=state.embodiment.physics.position.device,
        ),
    )

    observations = _assemble(state)

    assert torch.isfinite(observations).all()
    # tiny / 100 underflows in the output dtype; the direction does not.
    assert observations[0, 2].item() == 0.0
    assert observations[0, 3].item() == pytest.approx(1.0, abs=1e-6)
    assert observations[0, 4].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
def test_very_large_finite_velocity_with_zero_component_stays_finite(dtype):
    state = _make_episode(dtype=dtype)
    maximum = torch.finfo(dtype).max

    # The zero component is intentional: unsafe 0 * infinity intermediate
    # arithmetic during normalization/rotation must not create NaN.
    state = _set_velocity(state, [[maximum, 0.0]])
    observations = _assemble(state)

    assert torch.isfinite(observations).all()
    assert observations[0, 5].item() == pytest.approx(1.0)
    assert observations[0, 6].item() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("dtype", DTYPES)
def test_very_large_finite_acceleration_with_zero_component_stays_finite(dtype):
    state = _make_episode(dtype=dtype)
    maximum = torch.finfo(dtype).max
    state.embodiment.measured_acceleration.copy_(
        torch.tensor([[0.0, -maximum]], dtype=dtype)
    )

    observations = _assemble(state)

    assert torch.isfinite(observations).all()
    assert observations[0, 20].item() == pytest.approx(0.0, abs=1e-12)
    assert observations[0, 21].item() == pytest.approx(-1.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_extreme_vector_rotation_with_same_sign_component_sums(dtype):
    state = _make_episode(dtype=dtype)
    maximum = torch.finfo(dtype).max

    state = _set_angles(state, [[math.pi / 4]])
    state = _set_velocity(state, [[maximum, maximum]])
    state.embodiment.measured_acceleration.copy_(
        torch.tensor([[maximum, -maximum]], dtype=dtype)
    )

    observations = _assemble(state)

    assert torch.isfinite(observations).all()
    assert torch.all(observations[0, 5:7].abs() <= 1.0)
    assert torch.all(observations[0, 20:22].abs() <= 1.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_zero_velocity_and_acceleration_remain_exact_zero(dtype):
    state = _make_episode(dtype=dtype)
    state = _set_angles(state, [[2.7]])

    observations = _assemble(state)

    assert torch.equal(
        observations[0, 5:7],
        torch.zeros(2, dtype=dtype),
    )
    assert torch.equal(
        observations[0, 20:22],
        torch.zeros(2, dtype=dtype),
    )


# ---------------------------------------------------------------------------
# Real lidar integration, custom ranges, and dead-row canonicalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", DTYPES)
def test_actual_custom_range_lidar_is_assembled_without_renormalization(dtype):
    state = _make_episode(dtype=dtype)
    config = SensorConfig(lidar_range_m=5.0)

    lidar = sense_lidar(
        state.embodiment.physics,
        state.generated_world.world,
        config,
    )
    observations = assemble_observations(state, lidar)

    assert lidar.shape == (1, 8)
    assert torch.all((lidar >= 0.0) & (lidar <= 1.0))
    assert torch.equal(observations[:, 10:18], lidar)


def test_alive_lidar_values_are_preserved_exactly():
    state = _make_episode(dtype=torch.float64)
    lidar = torch.tensor(
        [[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0]],
        dtype=torch.float64,
    )

    observations = assemble_observations(state, lidar)
    assert torch.equal(observations[:, 10:18], lidar)


def test_dead_lidar_is_canonicalized_to_ones_even_if_input_differs():
    state = _make_episode(dtype=torch.float64)
    state = _set_lifecycle(state, alive=[[False]])

    lidar = torch.tensor(
        [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]],
        dtype=torch.float64,
    )
    observations = assemble_observations(state, lidar)

    assert torch.equal(
        observations[:, 10:18],
        torch.ones((1, 8), dtype=torch.float64),
    )
    assert torch.equal(
        observations[:, :10],
        torch.zeros((1, 10), dtype=torch.float64),
    )
    assert torch.equal(
        observations[:, 18:],
        torch.zeros((1, 5), dtype=torch.float64),
    )


@pytest.mark.parametrize(
    "bad_lidar",
    [
        [[-0.01] * 8],
        [[1.01] * 8],
        [[float("nan")] + [0.0] * 7],
        [[float("inf")] + [0.0] * 7],
    ],
)
def test_invalid_lidar_rejected_even_for_dead_agents(bad_lidar):
    state = _make_episode(dtype=torch.float64)
    state = _set_lifecycle(state, alive=[[False]])
    lidar = torch.tensor(bad_lidar, dtype=torch.float64)

    with pytest.raises((ValueError, TypeError)):
        assemble_observations(state, lidar)


# ---------------------------------------------------------------------------
# Lifecycle: physics.alive is the sole active mask
# ---------------------------------------------------------------------------

def test_extracted_agent_has_zero_non_lidar_channels():
    state = _make_episode(dtype=torch.float64)
    state = _set_lifecycle(
        state,
        alive=[[False]],
        extracted=[[True]],
        failed=[[False]],
    )
    observations = _assemble(state)

    assert torch.equal(
        observations[:, :10],
        torch.zeros((1, 10), dtype=torch.float64),
    )
    assert torch.equal(
        observations[:, 10:18],
        torch.ones((1, 8), dtype=torch.float64),
    )
    assert torch.equal(
        observations[:, 18:],
        torch.zeros((1, 5), dtype=torch.float64),
    )


def test_failed_agent_has_zero_non_lidar_channels():
    state = _make_episode(dtype=torch.float64)
    state = _set_lifecycle(
        state,
        alive=[[False]],
        extracted=[[False]],
        failed=[[True]],
    )
    observations = _assemble(state)

    assert torch.equal(
        observations[:, :10],
        torch.zeros((1, 10), dtype=torch.float64),
    )
    assert torch.equal(
        observations[:, 10:18],
        torch.ones((1, 8), dtype=torch.float64),
    )
    assert torch.equal(
        observations[:, 18:],
        torch.zeros((1, 5), dtype=torch.float64),
    )


def test_dead_without_terminal_annotation_is_still_inactive():
    state = _make_episode(dtype=torch.float64)
    state = _set_lifecycle(
        state,
        alive=[[False]],
        extracted=[[False]],
        failed=[[False]],
    )

    observations = _assemble(state)

    assert torch.equal(
        observations[:, :10],
        torch.zeros((1, 10), dtype=torch.float64),
    )
    assert torch.equal(
        observations[:, 10:18],
        torch.ones((1, 8), dtype=torch.float64),
    )


def test_energy_zero_with_alive_true_remains_active():
    state = _make_episode(dtype=torch.float64)
    state = _set_energy(state, [[0.0]])

    assert state.embodiment.physics.alive.item()
    observations = _assemble(state)

    assert observations[0, 4].item() > 0.0
    assert observations[0, 9].item() == pytest.approx(1.0)
    assert observations[0, 22].item() == 0.0


# ---------------------------------------------------------------------------
# Input ownership, output ownership, and immutability
# ---------------------------------------------------------------------------

def test_assembly_does_not_mutate_episode_or_lidar_inputs():
    state = _make_episode(dtype=torch.float64)
    lidar = torch.linspace(0.0, 1.0, 8, dtype=torch.float64).reshape(1, 8)

    before_position = state.embodiment.physics.position.clone()
    before_velocity = state.embodiment.physics.velocity.clone()
    before_angle = state.embodiment.physics.angle.clone()
    before_actions = state.embodiment.previous_actions.clone()
    before_acceleration = state.embodiment.measured_acceleration.clone()
    before_lidar = lidar.clone()
    before_collected = state.objectives.collected.clone()

    assemble_observations(state, lidar)

    assert torch.equal(state.embodiment.physics.position, before_position)
    assert torch.equal(state.embodiment.physics.velocity, before_velocity)
    assert torch.equal(state.embodiment.physics.angle, before_angle)
    assert torch.equal(state.embodiment.previous_actions, before_actions)
    assert torch.equal(
        state.embodiment.measured_acceleration,
        before_acceleration,
    )
    assert torch.equal(lidar, before_lidar)
    assert torch.equal(state.objectives.collected, before_collected)


def test_observation_output_does_not_alias_inputs_or_other_outputs():
    state = _make_episode(dtype=torch.float64)
    lidar = torch.full((1, 8), 0.5, dtype=torch.float64)

    first = assemble_observations(state, lidar)
    second = assemble_observations(state, lidar)

    assert first.data_ptr() != second.data_ptr()
    assert first.data_ptr() != state.embodiment.physics.position.data_ptr()
    assert first.data_ptr() != lidar.data_ptr()

    first[0, 10] = 0.0
    assert second[0, 10].item() == pytest.approx(0.5)
    assert lidar[0, 0].item() == pytest.approx(0.5)


def test_output_is_newly_allocated_in_original_dtype():
    for dtype in DTYPES:
        state = _make_episode(dtype=dtype)
        observations = _assemble(state)

        assert observations.dtype == dtype
        assert observations.device == state.embodiment.physics.position.device
        assert observations.data_ptr() != (
            state.embodiment.physics.position.data_ptr()
        )


# ---------------------------------------------------------------------------
# Strict validation, including malformed mutable physical fields
# ---------------------------------------------------------------------------

def test_non_tensor_position_rejected_before_batch_shape_metadata_access():
    state = _make_episode(dtype=torch.float64)

    # Deliberately corrupt mutable dataclass storage. The public batch_shape
    # property accesses position.shape, so the observation boundary must
    # check the position tensor type before reading batch_shape.
    object.__setattr__(state.embodiment.physics, "position", None)

    with pytest.raises(TypeError, match="physics.position must be a torch.Tensor"):
        assemble_observations(
            state,
            torch.full((1, 8), 0.5, dtype=torch.float64),
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "error_type", "message"),
    [
        (
            "velocity",
            None,
            TypeError,
            "physics.velocity must be a torch.Tensor",
        ),
        (
            "angle",
            torch.zeros((1, 2), dtype=torch.float64),
            ValueError,
            "physics.angle must have shape",
        ),
        (
            "angular_velocity",
            torch.zeros((1, 1), dtype=torch.float32),
            TypeError,
            "physics.angular_velocity must have dtype",
        ),
        (
            "energy",
            torch.tensor([[float("nan")]], dtype=torch.float64),
            ValueError,
            "physics.energy must contain only finite values",
        ),
    ],
)
def test_malformed_mutable_physics_fields_have_specific_errors(
    field,
    bad_value,
    error_type,
    message,
):
    state = _make_episode(dtype=torch.float64)
    object.__setattr__(state.embodiment.physics, field, bad_value)

    with pytest.raises(error_type, match=message):
        assemble_observations(
            state,
            torch.full((1, 8), 0.5, dtype=torch.float64),
        )


def test_lidar_shape_dtype_and_device_are_validated():
    state = _make_episode(dtype=torch.float64)

    with pytest.raises(ValueError, match="lidar_distances must have shape"):
        assemble_observations(
            state,
            torch.zeros((1, 7), dtype=torch.float64),
        )

    with pytest.raises(TypeError, match="lidar_distances must have dtype"):
        assemble_observations(
            state,
            torch.zeros((1, 8), dtype=torch.float32),
        )


def test_nonfinite_position_is_rejected():
    state = _make_episode(dtype=torch.float64)
    state.embodiment.physics.position[0, 0] = float("nan")

    with pytest.raises(
        ValueError,
        match="physics.position must contain only finite",
    ):
        _assemble(state)


def test_nonfinite_velocity_and_acceleration_are_rejected():
    state = _make_episode(dtype=torch.float64)
    state.embodiment.physics.velocity[0, 0] = float("inf")

    with pytest.raises(
        ValueError,
        match="physics.velocity must contain only finite",
    ):
        _assemble(state)

    state = _make_episode(dtype=torch.float64)
    state.embodiment.measured_acceleration[0, 1] = float("nan")

    with pytest.raises(
        ValueError,
        match="embodiment.measured_acceleration must contain only finite",
    ):
        _assemble(state)


# ---------------------------------------------------------------------------
# Batched/independent equivalence and device agreement
# ---------------------------------------------------------------------------

def test_batched_observations_equal_independent_agent_assembly():
    batch_shape = (2, 3)
    batched = _make_episode(
        dtype=torch.float64,
        batch_shape=batch_shape,
    )

    positions = torch.tensor(
        [
            [[10.0, 10.0], [20.0, 10.0], [30.0, 10.0]],
            [[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]],
        ],
        dtype=torch.float64,
    )
    batched = _set_positions(batched, positions)

    angles = torch.tensor(
        [
            [[0.0], [0.2], [0.4]],
            [[0.6], [0.8], [1.0]],
        ],
        dtype=torch.float64,
    )
    batched = _set_angles(batched, angles)

    lidar = torch.linspace(
        0.0,
        1.0,
        2 * 3 * 8,
        dtype=torch.float64,
    ).reshape(2, 3, 8)

    combined = assemble_observations(batched, lidar)
    assert combined.shape == (2, 3, 23)

    for row in range(2):
        for column in range(3):
            single = _make_episode(dtype=torch.float64)
            single = _set_positions(
                single,
                positions[row, column].reshape(1, 2),
            )
            single = _set_angles(
                single,
                angles[row, column].reshape(1, 1),
            )
            independent = assemble_observations(
                single,
                lidar[row, column].reshape(1, 8),
            )
            torch.testing.assert_close(
                combined[row, column],
                independent[0],
                rtol=1e-12,
                atol=1e-12,
            )


@pytest.mark.parametrize("dtype", DTYPES)
def test_cpu_cuda_numerical_agreement(dtype):
    if not _cuda_available():
        pytest.skip("CUDA is unavailable")

    cpu_state = _make_episode(dtype=dtype, device="cpu")
    cuda_state = _make_episode(dtype=dtype, device="cuda")

    cpu_state = _set_positions(cpu_state, [[17.0, 23.0]])
    cuda_state = _set_positions(cuda_state, [[17.0, 23.0]])

    cpu_state = _set_angles(cpu_state, [[0.37]])
    cuda_state = _set_angles(cuda_state, [[0.37]])

    cpu_state = _set_velocity(cpu_state, [[3.0, -7.0]])
    cuda_state = _set_velocity(cuda_state, [[3.0, -7.0]])

    cpu_state.embodiment.measured_acceleration.copy_(
        torch.tensor([[5.0, -9.0]], dtype=dtype)
    )
    cuda_state.embodiment.measured_acceleration.copy_(
        torch.tensor([[5.0, -9.0]], dtype=dtype, device="cuda")
    )

    lidar_cpu = torch.linspace(0.0, 1.0, 8, dtype=dtype).reshape(1, 8)
    lidar_cuda = lidar_cpu.to("cuda")

    cpu_output = assemble_observations(cpu_state, lidar_cpu)
    cuda_output = assemble_observations(cuda_state, lidar_cuda).cpu()

    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(
        cpu_output,
        cuda_output,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("device_name", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_observation_output_bounds_and_finiteness(dtype, device_name):
    if device_name == "cuda" and not _cuda_available():
        pytest.skip("CUDA is unavailable")

    state = _make_episode(dtype=dtype, device=device_name)
    observations = _assemble(state)

    assert torch.isfinite(observations).all()

    assert torch.all(observations[..., 0:2] >= -1.0)
    assert torch.all(observations[..., 0:2] <= 1.0)
    assert torch.all(observations[..., 2] >= 0.0)
    assert torch.all(observations[..., 2] <= 1.0)
    assert torch.all(observations[..., 3:5] >= -1.0)
    assert torch.all(observations[..., 3:5] <= 1.0)
    assert torch.all(observations[..., 5:8] >= -1.0)
    assert torch.all(observations[..., 5:8] <= 1.0)
    assert torch.all(observations[..., 10:18] >= 0.0)
    assert torch.all(observations[..., 10:18] <= 1.0)
    assert torch.all(observations[..., 18:20] >= -1.0)
    assert torch.all(observations[..., 18:20] <= 1.0)
    assert torch.all(observations[..., 20:22] >= -1.0)
    assert torch.all(observations[..., 20:22] <= 1.0)
    assert torch.all(observations[..., 22] >= 0.0)
    assert torch.all(observations[..., 22] <= 1.0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available",
            ),
        ),
    ],
)
def test_smallest_normal_displacement_has_positive_normalized_distance(
    dtype, device
):
    state = _make_episode(dtype=dtype, device=device)

    smallest_normal = torch.finfo(dtype).tiny
    maximum = torch.finfo(dtype).max
    device_obj = state.embodiment.physics.position.device

    state.generated_world.extraction_center.copy_(
        torch.tensor(
            [maximum, smallest_normal],
            dtype=dtype,
            device=device_obj,
        )
    )
    state = _set_positions(
        state,
        torch.tensor(
            [[maximum, 0.0]],
            dtype=dtype,
            device=device_obj,
        ),
    )

    observations = _assemble(state)

    assert torch.isfinite(observations).all()
    assert observations[0, 2].item() > 0.0
    assert observations[0, 3].item() == pytest.approx(1.0, abs=1e-6)
    assert observations[0, 4].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available",
            ),
        ),
    ],
)
def test_opposite_sign_near_maximum_axis_aligned_coordinates_do_not_overflow(
    dtype, device
):
    state = _make_episode(dtype=dtype, device=device)

    maximum = torch.finfo(dtype).max
    device_obj = state.embodiment.physics.position.device

    state.generated_world.extraction_center.copy_(
        torch.tensor(
            [maximum, 0.0],
            dtype=dtype,
            device=device_obj,
        )
    )
    state = _set_positions(
        state,
        torch.tensor(
            [[-maximum, 0.0]],
            dtype=dtype,
            device=device_obj,
        ),
    )

    observations = _assemble(state)

    assert torch.isfinite(observations).all()
    assert observations[0, 0].item() == pytest.approx(1.0)
    assert observations[0, 1].item() == pytest.approx(0.0)
    assert observations[0, 3].item() == pytest.approx(0.0, abs=1e-6)
    assert observations[0, 4].item() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", id="cpu"),
        pytest.param(
            "cuda",
            id="cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA is not available",
            ),
        ),
    ],
)
def test_near_maximum_velocity_and_acceleration_normalize_finitely(
    dtype, device
):
    state = _make_episode(dtype=dtype, device=device)

    maximum = torch.finfo(dtype).max
    device_obj = state.embodiment.physics.position.device

    state = _set_velocity(
        state,
        torch.tensor(
            [[maximum, 0.0]],
            dtype=dtype,
            device=device_obj,
        ),
    )
    state.embodiment.measured_acceleration.copy_(
        torch.tensor(
            [[maximum, 0.0]],
            dtype=dtype,
            device=device_obj,
        )
    )

    observations = _assemble(state)

    assert torch.isfinite(observations).all()
    assert observations[0, 5].item() == pytest.approx(1.0)
    assert observations[0, 6].item() == pytest.approx(0.0)
    assert observations[0, 20].item() == pytest.approx(1.0)
    assert observations[0, 21].item() == pytest.approx(0.0)


@pytest.mark.parametrize("dtype", DTYPES)
def test_mixed_alive_dead_non_square_batch_masks_all_23_channels(dtype):
    state = _make_episode(
        dtype=dtype,
        batch_shape=(2, 3),
        extraction_center=(80.0, 80.0),
    )
    device = state.embodiment.physics.position.device

    positions = torch.tensor(
        [
            [[10.0, 10.0], [20.0, 10.0], [30.0, 10.0]],
            [[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]],
        ],
        dtype=dtype,
        device=device,
    )
    state = _set_positions(state, positions)

    velocities = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ],
        dtype=dtype,
        device=device,
    )
    state = _set_velocity(state, velocities)

    angles = torch.tensor(
        [[[0.0], [0.1], [0.2]], [[0.3], [0.4], [0.5]]],
        dtype=dtype,
        device=device,
    )
    state = _set_angles(state, angles)

    angular_velocity = torch.tensor(
        [[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]],
        dtype=dtype,
        device=device,
    )
    state = _set_angular_velocity(state, angular_velocity)

    energy = torch.tensor(
        [[[100.0], [75.0], [50.0]], [[25.0], [0.0], [90.0]]],
        dtype=dtype,
        device=device,
    )
    state = _set_energy(state, energy)

    previous_actions = torch.tensor(
        [
            [[0.1, -0.1], [0.2, -0.2], [0.3, -0.3]],
            [[0.4, -0.4], [0.5, -0.5], [0.6, -0.6]],
        ],
        dtype=dtype,
        device=device,
    )
    state.embodiment.previous_actions.copy_(previous_actions)

    measured_acceleration = torch.tensor(
        [
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
            [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]],
        ],
        dtype=dtype,
        device=device,
    )
    state.embodiment.measured_acceleration.copy_(measured_acceleration)

    alive = torch.tensor(
        [
            [[True], [False], [True]],
            [[False], [True], [False]],
        ],
        dtype=torch.bool,
        device=device,
    )
    state = _set_lifecycle(state, alive=alive)

    lidar = torch.tensor(
        [
            [
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
                [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
            ],
            [
                [0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38],
                [0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47, 0.48],
                [0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58],
            ],
        ],
        dtype=dtype,
        device=device,
    )

    observations = _assemble(state, lidar)

    assert observations.shape == (2, 3, 23)
    assert torch.isfinite(observations).all()

    assert alive.shape == (2, 3, 1)
    assert state.embodiment.physics.angle.shape == (2, 3, 1)
    assert state.embodiment.physics.angular_velocity.shape == (2, 3, 1)
    assert state.embodiment.physics.energy.shape == (2, 3, 1)

    # Independent analytic reference for every alive agent.
    for b in range(2):
        for n in range(3):
            row = observations[b, n]

            if not alive[b, n, 0].item():
                expected_dead = torch.zeros(23, dtype=dtype, device=device)
                expected_dead[10:18] = 1.0

                torch.testing.assert_close(
                    row,
                    expected_dead,
                    atol=0.0,
                    rtol=0.0,
                )
                continue

            theta = angles[b, n, 0].item()
            c = math.cos(theta)
            s = math.sin(theta)

            dx = 80.0 - positions[b, n, 0].item()
            dy = 80.0 - positions[b, n, 1].item()
            distance = math.hypot(dx, dy)

            # Rotate world vectors into the current body frame.
            target_x_body = c * dx + s * dy
            target_y_body = -s * dx + c * dy

            velocity_x_body = (
                c * velocities[b, n, 0].item()
                + s * velocities[b, n, 1].item()
            )
            velocity_y_body = (
                -s * velocities[b, n, 0].item()
                + c * velocities[b, n, 1].item()
            )

            acceleration_x_body = (
                c * measured_acceleration[b, n, 0].item()
                + s * measured_acceleration[b, n, 1].item()
            )
            acceleration_y_body = (
                -s * measured_acceleration[b, n, 0].item()
                + c * measured_acceleration[b, n, 1].item()
            )

            body_distance = math.hypot(
                target_x_body,
                target_y_body,
            )

            expected = torch.zeros(23, dtype=dtype, device=device)

            expected[0] = max(-1.0, min(1.0, target_x_body / 50.0))
            expected[1] = max(-1.0, min(1.0, target_y_body / 50.0))
            expected[2] = min(distance / 100.0, 1.0)

            if body_distance == 0.0:
                expected[3] = 0.0
                expected[4] = 1.0
            else:
                expected[3] = target_y_body / body_distance
                expected[4] = target_x_body / body_distance

            expected[5] = max(
                -1.0, min(1.0, velocity_x_body / 12.0)
            )
            expected[6] = max(
                -1.0, min(1.0, velocity_y_body / 12.0)
            )
            expected[7] = max(
                -1.0,
                min(
                    1.0,
                    angular_velocity[b, n, 0].item() / 4.0,
                ),
            )
            expected[8] = math.sin(theta)
            expected[9] = math.cos(theta)

            expected[10:18] = lidar[b, n]
            expected[18:20] = previous_actions[b, n]

            expected[20] = max(
                -1.0, min(1.0, acceleration_x_body / 16.0)
            )
            expected[21] = max(
                -1.0, min(1.0, acceleration_y_body / 16.0)
            )
            expected[22] = energy[b, n, 0].item() / 100.0

            torch.testing.assert_close(
                row,
                expected,
                atol=2e-6,
                rtol=2e-6,
            )

    # Explicitly retain the alive, zero-energy agent.
    assert alive[1, 1, 0].item()
    assert energy[1, 1, 0].item() == 0.0
    assert observations[1, 1, 22].item() == pytest.approx(0.0)

    # The alive zero-energy row still retains its lidar and other observations.
    assert torch.equal(
        observations[1, 1, 10:18],
        lidar[1, 1],
    )
    assert observations[1, 1, 0:10].abs().sum().item() > 0.0


@pytest.mark.parametrize(
    ("bad_value", "message"),
    [
        (
            float("nan"),
            "previous_actions must contain only finite",
        ),
        (
            float("inf"),
            "previous_actions must contain only finite",
        ),
        (
            1.5,
            r"previous_actions must lie in \[-1, 1\]",
        ),
    ],
)
def test_malformed_previous_actions_are_rejected_at_observation_boundary(
    bad_value, message
):
    state = _make_episode(dtype=torch.float64)
    state.embodiment.previous_actions[0, 0] = bad_value

    with pytest.raises(ValueError, match=message):
        _assemble(state)


@pytest.mark.parametrize("field_name", ["alive", "extracted", "failed"])
def test_malformed_lifecycle_mask_dtype_is_rejected_at_observation_boundary(
    field_name
):
    state = _make_episode(dtype=torch.float64)

    if field_name == "alive":
        owner = state.embodiment.physics
    else:
        owner = state.embodiment

    original = getattr(owner, field_name)
    malformed = original.to(dtype=torch.float64)

    object.__setattr__(owner, field_name, malformed)

    with pytest.raises(
        (TypeError, ValueError),
        match=field_name,
    ):
        _assemble(state)
