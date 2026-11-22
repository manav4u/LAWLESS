"""Focused tests for LAWLESS P1.5 eight-ray lidar."""

import math

import pytest
import torch

from lawless.physics.state import AgentState
from lawless.sensors import sense_lidar
from lawless.world.config import SensorConfig
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState


def make_agent(
    position,
    angle=0.0,
    *,
    alive=True,
    dtype=torch.float64,
    device="cpu",
):
    position = torch.as_tensor(
        position,
        dtype=dtype,
        device=device,
    )

    if position.ndim < 1 or position.shape[-1] != 2:
        raise ValueError("test position must have shape [..., 2]")

    batch_shape = position.shape[:-1]

    angle = torch.as_tensor(
        angle,
        dtype=dtype,
        device=device,
    )

    if angle.numel() == 1:
        angle = angle.expand(batch_shape + (1,)).clone()
    elif angle.shape == batch_shape:
        angle = angle.unsqueeze(-1)

    alive = torch.as_tensor(
        alive,
        dtype=torch.bool,
        device=device,
    )

    if alive.numel() == 1:
        alive = alive.expand(batch_shape + (1,)).clone()
    elif alive.shape == batch_shape:
        alive = alive.unsqueeze(-1)

    return AgentState(
        position=position,
        velocity=torch.zeros_like(position),
        angle=angle,
        angular_velocity=torch.zeros_like(angle),
        energy=torch.full_like(angle, 100.0),
        alive=alive,
    )


def make_world(
    width=100.0,
    height=100.0,
    centers=(),
    radii=(),
    *,
    dtype=torch.float64,
    device="cpu",
):
    centers = torch.as_tensor(
        centers,
        dtype=dtype,
        device=device,
    ).reshape(-1, 2)

    radii = torch.as_tensor(
        radii,
        dtype=dtype,
        device=device,
    ).reshape(-1)

    return WorldState(
        width_m=width,
        height_m=height,
        obstacles=CircleObstacles(
            centers=centers,
            radii=radii,
        ),
    )


def make_config(max_range=20.0):
    return SensorConfig(lidar_range_m=max_range)


def test_output_shape_dtype_device_and_normalized_range():
    state = make_agent(
        [[50.0, 50.0], [25.0, 25.0]],
        angle=[[0.0], [math.pi / 4]],
    )
    world = make_world()

    result = sense_lidar(state, world, make_config())

    assert result.shape == (2, 8)
    assert result.dtype == state.position.dtype
    assert result.device == state.position.device
    assert torch.isfinite(result).all()
    assert torch.all(result >= 0.0)
    assert torch.all(result <= 1.0)
    assert torch.equal(result, torch.ones_like(result))


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
)
def test_ray_order_is_counterclockwise(dtype):
    origin = torch.tensor([50.0, 50.0], dtype=dtype)

    radial_distances = [
        3.0 + 1.5 * index
        for index in range(8)
    ]

    centers = []
    radii = []

    for index, distance in enumerate(radial_distances):
        theta = 2.0 * math.pi * index / 8.0

        centers.append(
            [
                50.0 + distance * math.cos(theta),
                50.0 + distance * math.sin(theta),
            ]
        )
        radii.append(0.2)

    state = make_agent(origin, dtype=dtype)
    world = make_world(
        centers=centers,
        radii=radii,
        dtype=dtype,
    )

    result = sense_lidar(state, world, make_config())

    expected = torch.tensor(
        [(distance - 0.2) / 20.0 for distance in radial_distances],
        dtype=dtype,
    )

    torch.testing.assert_close(
        result,
        expected,
        atol=2e-5,
        rtol=2e-5,
    )


def test_agent_rotation_rotates_world_space_ray_directions():
    world = make_world(
        centers=[[60.0, 50.0]],
        radii=[1.0],
    )

    heading_east = sense_lidar(
        make_agent([50.0, 50.0], angle=0.0),
        world,
        make_config(),
    )

    heading_north = sense_lidar(
        make_agent([50.0, 50.0], angle=math.pi / 2.0),
        world,
        make_config(),
    )

    assert heading_east[0].item() == pytest.approx(9.0 / 20.0)

    # Body-relative ray 6 points east after a 90-degree CCW rotation.
    assert heading_north[6].item() == pytest.approx(9.0 / 20.0)

    assert heading_north[0].item() == pytest.approx(1.0)


def test_nearest_obstacle_surface_wins_over_world_boundary():
    state = make_agent([5.0, 5.0])

    world = make_world(
        width=10.0,
        height=10.0,
        centers=[[7.0, 5.0]],
        radii=[0.5],
    )

    result = sense_lidar(state, world, make_config())

    # Circle surface is 1.5 m away; right world boundary is 5 m away.
    assert result[0].item() == pytest.approx(1.5 / 20.0)


def test_boundary_is_detected_when_no_obstacles_exist():
    state = make_agent([5.0, 5.0])

    result = sense_lidar(
        state,
        make_world(
            width=10.0,
            height=10.0,
        ),
        make_config(max_range=20.0),
    )

    expected = torch.tensor(
        [
            5.0 / 20.0,
            (5.0 * math.sqrt(2.0)) / 20.0,
            5.0 / 20.0,
            (5.0 * math.sqrt(2.0)) / 20.0,
            5.0 / 20.0,
            (5.0 * math.sqrt(2.0)) / 20.0,
            5.0 / 20.0,
            (5.0 * math.sqrt(2.0)) / 20.0,
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(result, expected)


def test_tangent_ray_counts_as_obstacle_hit():
    state = make_agent([5.0, 5.0])

    world = make_world(
        centers=[[10.0, 6.0]],
        radii=[1.0],
    )

    result = sense_lidar(state, world, make_config())

    assert result[0].item() == pytest.approx(5.0 / 20.0)


def test_near_tangent_miss_is_not_a_hit():
    state = make_agent([5.0, 5.0])

    world = make_world(
        centers=[[10.0, 6.01]],
        radii=[1.0],
    )

    result = sense_lidar(state, world, make_config())

    assert result[0].item() == pytest.approx(1.0)


def test_obstacle_behind_ray_is_ignored():
    state = make_agent([5.0, 5.0])

    world = make_world(
        centers=[[3.0, 5.0]],
        radii=[0.5],
    )

    result = sense_lidar(state, world, make_config())

    assert result[0].item() == pytest.approx(1.0)


def test_obstacle_beyond_maximum_range_is_ignored():
    state = make_agent([5.0, 5.0])

    world = make_world(
        centers=[[30.0, 5.0]],
        radii=[1.0],
    )

    result = sense_lidar(state, world, make_config(max_range=20.0))

    assert result[0].item() == pytest.approx(1.0)


def test_obstacle_inside_origin_reports_forward_exit_surface():
    state = make_agent([5.0, 5.0])

    world = make_world(
        centers=[[5.0, 5.0]],
        radii=[2.0],
    )

    result = sense_lidar(state, world, make_config())

    assert result[0].item() == pytest.approx(2.0 / 20.0)


def test_obstacle_surface_origin_reports_zero():
    state = make_agent([5.0, 5.0])

    world = make_world(
        centers=[[7.0, 5.0]],
        radii=[2.0],
    )

    result = sense_lidar(state, world, make_config())

    assert result[0].item() == pytest.approx(0.0)


def test_obstacle_surface_origin_pointing_outward_reports_zero():
    state = make_agent(
        [5.0, 5.0],
        angle=math.pi,
    )

    world = make_world(
        centers=[[7.0, 5.0]],
        radii=[2.0],
    )

    result = sense_lidar(state, world, make_config())

    assert result[0].item() == pytest.approx(0.0)


def test_edge_origin_outward_reports_zero():
    state = make_agent([0.0, 5.0], angle=math.pi)

    result = sense_lidar(
        state,
        make_world(width=10.0, height=10.0),
        make_config(),
    )

    assert result[0].item() == pytest.approx(0.0)


def test_edge_origin_inward_reaches_opposite_boundary():
    state = make_agent([0.0, 5.0], angle=0.0)

    result = sense_lidar(
        state,
        make_world(width=10.0, height=10.0),
        make_config(),
    )

    assert result[0].item() == pytest.approx(10.0 / 20.0)


def test_corner_origin_handles_outward_and_inward_directions():
    state = make_agent([0.0, 0.0])

    result = sense_lidar(
        state,
        make_world(width=10.0, height=10.0),
        make_config(),
    )

    assert result[4].item() == pytest.approx(0.0)

    expected_diagonal = (
        10.0 * math.sqrt(2.0) / 20.0
    )

    assert result[1].item() == pytest.approx(expected_diagonal)


def test_live_agent_outside_world_is_rejected():
    state = make_agent([-0.1, 5.0])

    with pytest.raises(ValueError, match="live agent"):
        sense_lidar(
            state,
            make_world(width=10.0, height=10.0),
            make_config(),
        )


def test_dead_agent_outside_world_returns_maximum_range_sentinels():
    state = make_agent(
        [-100.0, 500.0],
        alive=False,
    )

    result = sense_lidar(
        state,
        make_world(width=10.0, height=10.0),
        make_config(),
    )

    assert result.shape == (8,)
    assert torch.equal(result, torch.ones_like(result))


def test_dead_agent_inside_world_also_returns_maximum_range_sentinels():
    state = make_agent(
        [5.0, 5.0],
        alive=False,
    )

    world = make_world(
        centers=[[6.0, 5.0]],
        radii=[0.5],
    )

    result = sense_lidar(state, world, make_config())

    assert torch.equal(result, torch.ones_like(result))


def test_raw_scalar_batch_agent_state_is_supported():
    state = make_agent(
        [5.0, 5.0],
        angle=0.0,
    )

    assert state.batch_shape == torch.Size([])

    result = sense_lidar(
        state,
        make_world(width=10.0, height=10.0),
        make_config(),
    )

    assert result.shape == (8,)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
)
def test_batched_results_match_independent_scalar_results(dtype):
    positions = [
        [10.0, 20.0],
        [30.0, 40.0],
        [50.0, 50.0],
    ]

    angles = [
        0.0,
        math.pi / 4.0,
        math.pi / 2.0,
    ]

    state = make_agent(
        positions,
        angle=angles,
        dtype=dtype,
    )

    world = make_world(
        width=100.0,
        height=100.0,
        centers=[[15.0, 20.0], [30.0, 45.0]],
        radii=[1.0, 2.0],
        dtype=dtype,
    )

    batched = sense_lidar(state, world, make_config())

    assert batched.shape == (3, 8)

    for index in range(3):
        scalar = sense_lidar(
            make_agent(
                positions[index],
                angle=angles[index],
                dtype=dtype,
            ),
            world,
            make_config(),
        )

        torch.testing.assert_close(
            batched[index],
            scalar,
        )


def test_multidimensional_batch_shape_is_preserved():
    state = make_agent(
        [
            [[10.0, 10.0], [20.0, 20.0]],
            [[30.0, 30.0], [40.0, 40.0]],
        ],
        dtype=torch.float64,
    )

    result = sense_lidar(
        state,
        make_world(),
        make_config(),
    )

    assert result.shape == (2, 2, 8)


def test_invalid_ray_count_is_rejected():
    state = make_agent([50.0, 50.0])

    with pytest.raises(ValueError, match="ray_count"):
        sense_lidar(
            state,
            make_world(),
            SensorConfig(lidar_ray_count=7),
        )


def test_nonpositive_lidar_range_is_rejected():
    state = make_agent([50.0, 50.0])

    with pytest.raises(ValueError, match="lidar_range"):
        sense_lidar(
            state,
            make_world(),
            SensorConfig(lidar_range_m=0.0),
        )


def test_nonfinite_position_is_rejected():
    state = make_agent([50.0, 50.0])
    state.position[0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        sense_lidar(state, make_world(), make_config())


def test_nonfinite_angle_is_rejected():
    state = make_agent([50.0, 50.0])
    state.angle[0] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        sense_lidar(state, make_world(), make_config())


def test_incompatible_obstacle_dtype_is_rejected():
    state = make_agent(
        [50.0, 50.0],
        dtype=torch.float32,
    )

    world = make_world(
        dtype=torch.float64,
    )

    with pytest.raises(TypeError, match="dtype"):
        sense_lidar(state, world, make_config())


def test_incompatible_obstacle_device_is_rejected():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    state = make_agent(
        [50.0, 50.0],
        dtype=torch.float32,
        device="cpu",
    )

    world = make_world(
        dtype=torch.float32,
        device="cuda",
    )

    with pytest.raises(ValueError, match="device"):
        sense_lidar(state, world, make_config())


def test_sensing_does_not_mutate_agent_or_world():
    state = make_agent(
        [[50.0, 50.0], [20.0, 30.0]],
        angle=[0.0, math.pi / 4.0],
    )

    world = make_world(
        centers=[[55.0, 50.0]],
        radii=[1.0],
    )

    state_before = {
        name: getattr(state, name).clone()
        for name in state.__dataclass_fields__
    }

    centers_before = world.obstacles.centers.clone()
    radii_before = world.obstacles.radii.clone()

    sense_lidar(state, world, make_config())

    for name, expected in state_before.items():
        assert torch.equal(
            getattr(state, name),
            expected,
        )

    assert torch.equal(
        world.obstacles.centers,
        centers_before,
    )

    assert torch.equal(
        world.obstacles.radii,
        radii_before,
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
)
def test_genuine_cuda_parity(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    cpu_state = make_agent(
        [
            [50.0, 50.0],
            [20.0, 30.0],
        ],
        angle=[
            0.0,
            math.pi / 4.0,
        ],
        dtype=dtype,
        device="cpu",
    )

    cuda_state = make_agent(
        [
            [50.0, 50.0],
            [20.0, 30.0],
        ],
        angle=[
            0.0,
            math.pi / 4.0,
        ],
        dtype=dtype,
        device="cuda",
    )

    cpu_world = make_world(
        centers=[
            [55.0, 50.0],
            [20.0, 35.0],
        ],
        radii=[1.0, 2.0],
        dtype=dtype,
        device="cpu",
    )

    cuda_world = make_world(
        centers=[
            [55.0, 50.0],
            [20.0, 35.0],
        ],
        radii=[1.0, 2.0],
        dtype=dtype,
        device="cuda",
    )

    cpu_result = sense_lidar(
        cpu_state,
        cpu_world,
        make_config(),
    )

    cuda_result = sense_lidar(
        cuda_state,
        cuda_world,
        make_config(),
    )

    assert cuda_result.device.type == "cuda"
    assert cuda_result.dtype == dtype
    assert cuda_result.shape == (2, 8)

    torch.testing.assert_close(
        cuda_result.cpu(),
        cpu_result,
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
)
def test_exact_tangent_and_adjacent_representable_clearances(dtype):
    tangent_y = torch.tensor(11.0, dtype=dtype)

    below_y = torch.nextafter(
        tangent_y,
        torch.tensor(float("-inf"), dtype=dtype),
    )

    above_y = torch.nextafter(
        tangent_y,
        torch.tensor(float("inf"), dtype=dtype),
    )

    assert below_y.item() < tangent_y.item()
    assert above_y.item() > tangent_y.item()

    def east_ray_reading(circle_y):
        state = make_agent(
            [10.0, 10.0],
            angle=0.0,
            dtype=dtype,
        )

        world = make_world(
            width=100.0,
            height=100.0,
            centers=[[20.0, circle_y.item()]],
            radii=[1.0],
            dtype=dtype,
        )

        return sense_lidar(
            state,
            world,
            make_config(max_range=20.0),
        )[0]

    # Exact tangent: center perpendicular distance is exactly 1 m.
    tangent = east_ray_reading(tangent_y)

    assert tangent.item() == pytest.approx(0.5)

    # One representable value below tangent: strict intersection.
    inside = east_ray_reading(below_y)

    assert inside.item() < 0.5
    assert inside.item() > 0.0

    # One representable value above tangent: strict miss.
    outside = east_ray_reading(above_y)

    assert outside.item() == 1.0

@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64],
)
def test_cuda_tangent_adjacent_representable_clearances(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is unavailable")

    device = "cuda"

    tangent_y = torch.tensor(
        11.0,
        dtype=dtype,
        device=device,
    )

    below_y = torch.nextafter(
        tangent_y,
        torch.tensor(float("-inf"), dtype=dtype, device=device),
    )

    above_y = torch.nextafter(
        tangent_y,
        torch.tensor(float("inf"), dtype=dtype, device=device),
    )

    def reading(circle_y):
        state = make_agent(
            [10.0, 10.0],
            angle=0.0,
            dtype=dtype,
            device=device,
        )

        world = make_world(
            centers=[
                [20.0, circle_y.item()],
            ],
            radii=[1.0],
            dtype=dtype,
            device=device,
        )

        return sense_lidar(
            state,
            world,
            make_config(),
        )[0]

    tangent = reading(tangent_y)
    inside = reading(below_y)
    outside = reading(above_y)

    assert tangent.device.type == "cuda"
    assert tangent.item() == pytest.approx(0.5)
    assert inside.item() < 0.5
    assert inside.item() > 0.0
    assert outside.item() == 1.0

def test_mutated_negative_obstacle_radius_is_rejected():
    state = make_agent([10.0, 10.0])

    world = make_world(
        centers=[[20.0, 10.0]],
        radii=[1.0],
    )

    world.obstacles.radii[0] = -1.0

    with pytest.raises(ValueError, match="strictly positive"):
        sense_lidar(state, world, make_config())


def test_mutated_zero_obstacle_radius_is_rejected():
    state = make_agent([10.0, 10.0])

    world = make_world(
        centers=[[20.0, 10.0]],
        radii=[1.0],
    )

    world.obstacles.radii[0] = 0.0

    with pytest.raises(ValueError, match="strictly positive"):
        sense_lidar(state, world, make_config())


def test_mutated_nonfinite_obstacle_radius_is_rejected():
    state = make_agent([10.0, 10.0])

    world = make_world(
        centers=[[20.0, 10.0]],
        radii=[1.0],
    )

    world.obstacles.radii[0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        sense_lidar(state, world, make_config())


def test_mutated_nonfinite_obstacle_center_is_rejected():
    state = make_agent([10.0, 10.0])

    world = make_world(
        centers=[[20.0, 10.0]],
        radii=[1.0],
    )

    world.obstacles.centers[0, 0] = float("inf")

    with pytest.raises(ValueError, match="finite"):
        sense_lidar(state, world, make_config())


def test_mutated_obstacle_outside_world_is_rejected():
    state = make_agent([10.0, 10.0])

    world = make_world(
        width=30.0,
        height=30.0,
        centers=[[20.0, 10.0]],
        radii=[1.0],
    )

    # Center remains finite, radius remains positive, but the circle
    # no longer fits inside the rectangular world.
    world.obstacles.centers[0, 0] = 29.5

    with pytest.raises(ValueError, match="fully contained"):
        sense_lidar(state, world, make_config())


def test_mutated_obstacle_tensor_shape_is_rejected():
    state = make_agent([10.0, 10.0])

    world = make_world(
        centers=[[20.0, 10.0]],
        radii=[1.0],
    )

    world.obstacles.centers.resize_(1, 3)

    with pytest.raises(ValueError, match="shape"):
        sense_lidar(state, world, make_config())


def test_non_tensor_agent_state_field_is_rejected_intentionally():
    state = make_agent([10.0, 10.0])

    state.angle = 0.0

    with pytest.raises(TypeError, match="state.angle"):
        sense_lidar(state, make_world(), make_config())


# P1.5 precision regressions. Append only; retain all existing tests.

import pytest
import torch

from lawless.physics.state import AgentState
from lawless.sensors.lidar import sense_lidar
from lawless.world.config import SensorConfig
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState


def _p15_precision_state(position, heading, *, dtype, device):
    position_tensor = torch.tensor(
        [position], dtype=dtype, device=device
    )
    return AgentState(
        position=position_tensor,
        velocity=torch.zeros_like(position_tensor),
        angle=torch.tensor([[heading]], dtype=dtype, device=device),
        angular_velocity=torch.zeros((1, 1), dtype=dtype, device=device),
        energy=torch.ones((1, 1), dtype=dtype, device=device),
        alive=torch.ones((1, 1), dtype=torch.bool, device=device),
    )


def _p15_precision_world(*, dtype, device, obstacle=None):
    if obstacle is None:
        centers = torch.empty((0, 2), dtype=dtype, device=device)
        radii = torch.empty((0,), dtype=dtype, device=device)
    else:
        center, radius = obstacle
        centers = torch.tensor([center], dtype=dtype, device=device)
        radii = torch.tensor([radius], dtype=dtype, device=device)

    return WorldState(
        width_m=100.0,
        height_m=100.0,
        obstacles=CircleObstacles(centers=centers, radii=radii),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("position", "heading", "expected_east"),
    [
        # Exactly parallel to the bottom edge: no immediate boundary hit.
        ((50.0, 0.0), 0.0, 1.0),
        # A nonzero outward heading exits immediately.
        ((50.0, 0.0), -5.0e-7, 0.0),
        ((50.0, 0.0), -1.0e-16, 0.0),
        # A nonzero inward heading does not hit the bottom edge.
        ((50.0, 0.0), 5.0e-7, 1.0),
        ((50.0, 0.0), 1.0e-16, 1.0),
        # Repeat the exact/adjacent distinction at the bottom-left corner.
        ((0.0, 0.0), 0.0, 1.0),
        ((0.0, 0.0), -5.0e-7, 0.0),
        ((0.0, 0.0), 5.0e-7, 1.0),
    ],
)
def test_small_nonzero_headings_are_not_snapped_at_edges_or_corners(
    dtype, position, heading, expected_east
):
    device = torch.device("cpu")
    state = _p15_precision_state(
        position, heading, dtype=dtype, device=device
    )
    world = _p15_precision_world(dtype=dtype, device=device)

    result = sense_lidar(state, world, SensorConfig(lidar_range_m=20.0))

    assert result.dtype == dtype
    assert result.device == device
    assert result.shape == (1, 8)
    assert result[0, 0].item() == pytest.approx(expected_east, abs=0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("heading", "center"),
    [
        (
            0.8236643671989441,
            (56.06173324584961, 58.01594543457031),
        ),
        (
            0.7453388571739197,
            (56.67036056518555, 57.517066955566406),
        ),
    ],
)
def test_oblique_strict_near_tangent_misses_are_not_reported_as_hits(
    dtype, heading, center
):
    device = torch.device("cpu")
    state = _p15_precision_state(
        (50.0, 50.0), heading, dtype=dtype, device=device
    )
    world = _p15_precision_world(
        dtype=dtype,
        device=device,
        obstacle=(center, 1.0),
    )

    result = sense_lidar(state, world, SensorConfig(lidar_range_m=20.0))

    assert result.dtype == dtype
    assert result.device == device
    assert result[0, 0].item() == 1.0

def test_lidar_rejects_unsupported_agent_geometry_dtype():
    device = torch.device("cpu")
    state = _p15_precision_state(
        (50.0, 50.0),
        0.0,
        dtype=torch.float16,
        device=device,
    )
    world = _p15_precision_world(dtype=torch.float32, device=device)

    with pytest.raises(TypeError, match="float32 or torch.float64"):
        sense_lidar(state, world, SensorConfig(lidar_range_m=20.0))

# P1.5 batch-mask and public keyword regressions. Append only.

import pytest
import torch

from lawless.physics.state import AgentState
from lawless.sensors.lidar import sense_lidar
from lawless.world.config import SensorConfig
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState


def _p15_batch_state(positions, headings, alive, *, dtype, device):
    position = torch.tensor(positions, dtype=dtype, device=device)
    batch_shape = position.shape[:-1]

    angle = torch.tensor(headings, dtype=dtype, device=device).reshape(
        batch_shape + (1,)
    )
    alive_tensor = torch.tensor(
        alive, dtype=torch.bool, device=device
    ).reshape(batch_shape + (1,))

    return AgentState(
        position=position,
        velocity=torch.zeros_like(position),
        angle=angle,
        angular_velocity=torch.zeros_like(angle),
        energy=torch.ones_like(angle),
        alive=alive_tensor,
    )


def _p15_empty_world(*, dtype, device):
    return WorldState(
        width_m=100.0,
        height_m=100.0,
        obstacles=CircleObstacles(
            centers=torch.empty((0, 2), dtype=dtype, device=device),
            radii=torch.empty((0,), dtype=dtype, device=device),
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lidar_supports_non_square_multidimensional_live_batch(dtype):
    device = torch.device("cpu")
    positions = [
        [[50.0, 50.0], [50.0, 50.0], [50.0, 50.0]],
        [[50.0, 50.0], [50.0, 50.0], [50.0, 50.0]],
    ]
    headings = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    alive = [[True, True, True], [True, True, True]]

    state = _p15_batch_state(
        positions, headings, alive, dtype=dtype, device=device
    )
    world = _p15_empty_world(dtype=dtype, device=device)

    result = sense_lidar(
        state=state,
        world=world,
        config=SensorConfig(lidar_range_m=20.0),
    )

    assert result.shape == (2, 3, 8)
    assert result.dtype == dtype
    assert result.device == device
    assert torch.equal(result, torch.ones_like(result))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lidar_excludes_dead_outside_agent_from_live_world_validation(dtype):
    device = torch.device("cpu")
    state = _p15_batch_state(
        positions=[[50.0, 50.0], [-1.0, 50.0]],
        headings=[0.0, 0.0],
        alive=[True, False],
        dtype=dtype,
        device=device,
    )
    world = _p15_empty_world(dtype=dtype, device=device)

    result = sense_lidar(
        state=state,
        world=world,
        config=SensorConfig(lidar_range_m=20.0),
    )

    assert result.shape == (2, 8)
    assert result.dtype == dtype
    assert result.device == device
    assert torch.equal(result, torch.ones_like(result))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lidar_supports_non_square_multidimensional_live_batch_cuda(dtype):
    device = torch.device("cuda")
    positions = [
        [[50.0, 50.0], [50.0, 50.0], [50.0, 50.0]],
        [[50.0, 50.0], [50.0, 50.0], [50.0, 50.0]],
    ]
    headings = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    alive = [[True, True, True], [True, True, True]]

    state = _p15_batch_state(
        positions, headings, alive, dtype=dtype, device=device
    )
    world = _p15_empty_world(dtype=dtype, device=device)

    result = sense_lidar(
        state=state,
        world=world,
        config=SensorConfig(lidar_range_m=20.0),
    )

    assert result.shape == (2, 3, 8)
    assert result.dtype == dtype
    assert result.device.type == "cuda"
    assert torch.equal(result, torch.ones_like(result))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lidar_excludes_dead_outside_agent_from_live_world_validation_cuda(dtype):
    device = torch.device("cuda")
    state = _p15_batch_state(
        positions=[[50.0, 50.0], [-1.0, 50.0]],
        headings=[0.0, 0.0],
        alive=[True, False],
        dtype=dtype,
        device=device,
    )
    world = _p15_empty_world(dtype=dtype, device=device)

    result = sense_lidar(
        state=state,
        world=world,
        config=SensorConfig(lidar_range_m=20.0),
    )

    assert result.shape == (2, 8)
    assert result.dtype == dtype
    assert result.device.type == "cuda"
    assert torch.equal(result, torch.ones_like(result))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("position", "heading", "expected_east"),
    [
        ((50.0, 0.0), 0.0, 1.0),
        ((50.0, 0.0), -5.0e-7, 0.0),
        ((50.0, 0.0), -1.0e-16, 0.0),
        ((50.0, 0.0), 5.0e-7, 1.0),
        ((50.0, 0.0), 1.0e-16, 1.0),
        ((0.0, 0.0), 0.0, 1.0),
        ((0.0, 0.0), -5.0e-7, 0.0),
        ((0.0, 0.0), 5.0e-7, 1.0),
    ],
)
def test_small_heading_edge_and_corner_regressions_cuda(
    dtype, position, heading, expected_east
):
    device = torch.device("cuda")
    state = _p15_precision_state(
        position, heading, dtype=dtype, device=device
    )
    world = _p15_precision_world(dtype=dtype, device=device)

    result = sense_lidar(state, world, SensorConfig(lidar_range_m=20.0))

    assert result.device.type == "cuda"
    assert result.dtype == dtype
    assert result.shape == (1, 8)
    assert result[0, 0].item() == pytest.approx(expected_east, abs=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("heading", "center"),
    [
        (
            0.8236643671989441,
            (56.06173324584961, 58.01594543457031),
        ),
        (
            0.7453388571739197,
            (56.67036056518555, 57.517066955566406),
        ),
    ],
)
def test_oblique_strict_near_tangent_misses_cuda(dtype, heading, center):
    device = torch.device("cuda")
    state = _p15_precision_state(
        (50.0, 50.0), heading, dtype=dtype, device=device
    )
    world = _p15_precision_world(
        dtype=dtype,
        device=device,
        obstacle=(center, 1.0),
    )

    result = sense_lidar(state, world, SensorConfig(lidar_range_m=20.0))

    assert result.device.type == "cuda"
    assert result.dtype == dtype
    assert result.shape == (1, 8)
    assert result[0, 0].item() == 1.0


# P1.5 heterogeneous multi-obstacle and custom-range regressions.
# Append only; retain all existing tests.

def _p15_heterogeneous_obstacle_world(*, dtype, device, reverse=False):
    centers = [
        [20.0, 50.0],
        [30.0, 50.0],
    ]
    radii = [2.0, 13.0]

    if reverse:
        centers.reverse()
        radii.reverse()

    return WorldState(
        width_m=100.0,
        height_m=100.0,
        obstacles=CircleObstacles(
            centers=torch.tensor(centers, dtype=dtype, device=device),
            radii=torch.tensor(radii, dtype=dtype, device=device),
        ),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reverse", [False, True])
def test_lidar_uses_nearest_surface_across_heterogeneous_circles_in_batch(
    dtype, reverse
):
    device = torch.device("cpu")

    positions = [
        [[10.0, 50.0], [10.0, 50.0], [10.0, 50.0]],
        [[10.0, 50.0], [10.0, 50.0], [10.0, 50.0]],
    ]
    headings = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    alive = [[True, True, True], [True, True, True]]

    state = _p15_batch_state(
        positions, headings, alive, dtype=dtype, device=device
    )
    world = _p15_heterogeneous_obstacle_world(
        dtype=dtype, device=device, reverse=reverse
    )

    result = sense_lidar(
        state=state,
        world=world,
        config=SensorConfig(lidar_range_m=10.0),
    )

    assert result.shape == (2, 3, 8)
    assert result.dtype == dtype
    assert result.device == device

    expected_east = torch.full(
        (2, 3), 0.7, dtype=dtype, device=device
    )
    torch.testing.assert_close(
        result[..., 0],
        expected_east,
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reverse", [False, True])
def test_lidar_uses_nearest_surface_across_heterogeneous_circles_in_batch_cuda(
    dtype, reverse
):
    device = torch.device("cuda")

    positions = [
        [[10.0, 50.0], [10.0, 50.0], [10.0, 50.0]],
        [[10.0, 50.0], [10.0, 50.0], [10.0, 50.0]],
    ]
    headings = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    alive = [[True, True, True], [True, True, True]]

    state = _p15_batch_state(
        positions, headings, alive, dtype=dtype, device=device
    )
    world = _p15_heterogeneous_obstacle_world(
        dtype=dtype, device=device, reverse=reverse
    )

    result = sense_lidar(
        state=state,
        world=world,
        config=SensorConfig(lidar_range_m=10.0),
    )

    assert result.shape == (2, 3, 8)
    assert result.dtype == dtype
    assert result.device.type == "cuda"

    expected_east = torch.full(
        (2, 3), 0.7, dtype=dtype, device=device
    )
    torch.testing.assert_close(
        result[..., 0],
        expected_east,
        rtol=1e-6,
        atol=1e-7,
    )

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("lidar_range", "expected"),
    [
        (5.0, 1.0),
        (10.0, 0.7),
        (30.0, 7.0 / 30.0),
    ],
)
def test_lidar_custom_range_normalization_and_clipping(
    dtype, lidar_range, expected
):
    device = torch.device("cpu")
    state = _p15_precision_state(
        (10.0, 50.0), 0.0, dtype=dtype, device=device
    )
    world = _p15_heterogeneous_obstacle_world(
        dtype=dtype, device=device
    )

    result = sense_lidar(
        state=state,
        world=world,
        config=SensorConfig(lidar_range_m=lidar_range),
    )

    assert result.shape == (1, 8)
    assert result.dtype == dtype
    assert result.device == device
    assert result[0, 0].item() == pytest.approx(
        expected,
        rel=1e-6,
        abs=1e-7,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    ("lidar_range", "expected"),
    [
        (5.0, 1.0),
        (10.0, 0.7),
        (30.0, 7.0 / 30.0),
    ],
)
def test_lidar_custom_range_normalization_and_clipping_cuda(
    dtype, lidar_range, expected
):
    device = torch.device("cuda")
    state = _p15_precision_state(
        (10.0, 50.0), 0.0, dtype=dtype, device=device
    )
    world = _p15_heterogeneous_obstacle_world(
        dtype=dtype, device=device
    )

    result = sense_lidar(
        state=state,
        world=world,
        config=SensorConfig(lidar_range_m=lidar_range),
    )

    assert result.shape == (1, 8)
    assert result.dtype == dtype
    assert result.device.type == "cuda"
    assert result[0, 0].item() == pytest.approx(
        expected,
        rel=1e-6,
        abs=1e-7,
    )


# P1.5 exact oblique surface-contact regressions. Append only.

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "heading",
    [
        0.0,
        0.3,
        0.7453388571739197,
        0.8236643671989441,
        -1.2,
        2.4,
    ],
)
def test_lidar_exact_3_4_5_surface_contact_is_zero_for_all_rays_cpu(
    dtype, heading
):
    device = torch.device("cpu")
    state = _p15_precision_state(
        (53.0, 54.0), heading, dtype=dtype, device=device
    )
    world = _p15_precision_world(
        dtype=dtype,
        device=device,
        obstacle=((50.0, 50.0), 5.0),
    )

    result = sense_lidar(
        state, world, SensorConfig(lidar_range_m=20.0)
    )

    assert result.shape == (1, 8)
    assert result.dtype == dtype
    assert result.device == device
    assert torch.equal(result, torch.zeros_like(result))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "heading",
    [
        0.0,
        0.3,
        0.7453388571739197,
        0.8236643671989441,
        -1.2,
        2.4,
    ],
)
def test_lidar_adjacent_surface_positions_are_not_collapsed_to_contact_cpu(
    dtype, heading
):
    device = torch.device("cpu")

    x = torch.tensor(53.0, dtype=dtype, device=device)
    toward_center = torch.tensor(50.0, dtype=dtype, device=device)
    away_from_center = torch.tensor(60.0, dtype=dtype, device=device)

    interior_x = torch.nextafter(x, toward_center).item()
    exterior_x = torch.nextafter(x, away_from_center).item()

    world = _p15_precision_world(
        dtype=dtype,
        device=device,
        obstacle=((50.0, 50.0), 5.0),
    )

    interior_state = _p15_precision_state(
        (interior_x, 54.0), heading, dtype=dtype, device=device
    )
    exterior_state = _p15_precision_state(
        (exterior_x, 54.0), heading, dtype=dtype, device=device
    )

    interior_result = sense_lidar(
        interior_state, world, SensorConfig(lidar_range_m=20.0)
    )
    exterior_result = sense_lidar(
        exterior_state, world, SensorConfig(lidar_range_m=20.0)
    )

    assert torch.all(interior_result > 0.0)
    assert torch.all(exterior_result > 0.0)
    assert not torch.any(interior_result == 0.0)
    assert not torch.any(exterior_result == 0.0)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "heading",
    [
        0.0,
        0.3,
        0.7453388571739197,
        0.8236643671989441,
        -1.2,
        2.4,
    ],
)
def test_lidar_exact_3_4_5_surface_contact_is_zero_for_all_rays_cuda(
    dtype, heading
):
    device = torch.device("cuda")
    state = _p15_precision_state(
        (53.0, 54.0), heading, dtype=dtype, device=device
    )
    world = _p15_precision_world(
        dtype=dtype,
        device=device,
        obstacle=((50.0, 50.0), 5.0),
    )

    result = sense_lidar(
        state, world, SensorConfig(lidar_range_m=20.0)
    )

    assert result.shape == (1, 8)
    assert result.dtype == dtype
    assert result.device.type == "cuda"
    assert torch.equal(result, torch.zeros_like(result))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "heading",
    [
        0.0,
        0.3,
        0.7453388571739197,
        0.8236643671989441,
        -1.2,
        2.4,
    ],
)
def test_lidar_adjacent_surface_positions_are_not_collapsed_to_contact_cuda(
    dtype, heading
):
    device = torch.device("cuda")

    x = torch.tensor(53.0, dtype=dtype, device=device)
    toward_center = torch.tensor(50.0, dtype=dtype, device=device)
    away_from_center = torch.tensor(60.0, dtype=dtype, device=device)

    interior_x = torch.nextafter(x, toward_center).item()
    exterior_x = torch.nextafter(x, away_from_center).item()

    world = _p15_precision_world(
        dtype=dtype,
        device=device,
        obstacle=((50.0, 50.0), 5.0),
    )

    interior_state = _p15_precision_state(
        (interior_x, 54.0), heading, dtype=dtype, device=device
    )
    exterior_state = _p15_precision_state(
        (exterior_x, 54.0), heading, dtype=dtype, device=device
    )

    interior_result = sense_lidar(
        interior_state, world, SensorConfig(lidar_range_m=20.0)
    )
    exterior_result = sense_lidar(
        exterior_state, world, SensorConfig(lidar_range_m=20.0)
    )

    assert interior_result.device.type == "cuda"
    assert exterior_result.device.type == "cuda"
    assert torch.all(interior_result > 0.0)
    assert torch.all(exterior_result > 0.0)
    assert not torch.any(interior_result == 0.0)
    assert not torch.any(exterior_result == 0.0)
