import pytest
import torch

from lawless.world.config import WorldConfig
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState


def make_obstacles(
    centers,
    radii,
    *,
    dtype=torch.float32,
    device="cpu",
):
    return CircleObstacles(
        centers=torch.tensor(centers, dtype=dtype, device=device).reshape(-1, 2),
        radii=torch.tensor(radii, dtype=dtype, device=device),
    )


def test_construct_single_world_with_contained_obstacles():
    obstacles = make_obstacles(
        [[2.0, 3.0], [97.0, 95.0]],
        [1.0, 3.0],
    )

    world = WorldState(
        width_m=100.0,
        height_m=100.0,
        obstacles=obstacles,
    )

    assert world.width_m == 100.0
    assert world.height_m == 100.0
    assert world.obstacle_count == 2
    assert world.obstacles is obstacles


def test_zero_obstacle_world_is_valid():
    obstacles = CircleObstacles.empty(dtype=torch.float64)

    world = WorldState(
        width_m=100.0,
        height_m=100.0,
        obstacles=obstacles,
    )

    positions = torch.tensor(
        [[0.0, 0.0], [50.0, 50.0], [100.0, 100.0]],
        dtype=torch.float64,
    )

    clearance = world.obstacle_clearance(positions)

    assert world.obstacle_count == 0
    assert clearance.shape == (3, 0)
    assert clearance.dtype == torch.float64
    assert clearance.device == positions.device


@pytest.mark.parametrize(
    "centers,radii",
    [
        ([[-0.1, 5.0]], [0.1]),
        ([[10.1, 5.0]], [0.2]),
        ([[5.0, -0.1]], [0.1]),
        ([[5.0, 10.1]], [0.2]),
        ([[0.0, 5.0]], [0.1]),
        ([[10.0, 5.0]], [0.1]),
        ([[5.0, 0.0]], [0.1]),
        ([[5.0, 10.0]], [0.1]),
    ],
)
def test_rejects_obstacle_circle_not_fully_contained(centers, radii):
    obstacles = make_obstacles(centers, radii)

    with pytest.raises(ValueError, match="fully contained"):
        WorldState(
            width_m=10.0,
            height_m=10.0,
            obstacles=obstacles,
        )


def test_obstacle_circle_may_be_tangent_to_world_edges():
    obstacles = make_obstacles(
        [
            [1.0, 5.0],
            [9.0, 5.0],
            [5.0, 1.0],
            [5.0, 9.0],
        ],
        [1.0, 1.0, 1.0, 1.0],
    )

    world = WorldState(
        width_m=10.0,
        height_m=10.0,
        obstacles=obstacles,
    )

    assert world.obstacle_count == 4


def test_contains_positions_uses_inclusive_world_edges():
    world = WorldState(
        width_m=10.0,
        height_m=20.0,
        obstacles=CircleObstacles.empty(),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 20.0],
            [5.0, 10.0],
            [-0.001, 10.0],
            [10.001, 10.0],
        ]
    )

    assert world.contains_positions(positions).tolist() == [
        True, True, True, False, False
    ]


def test_world_boundary_clearance_uses_minimum_signed_axis_clearance():
    world = WorldState(
        width_m=10.0,
        height_m=10.0,
        obstacles=CircleObstacles.empty(),
    )
    positions = torch.tensor(
        [
            [5.0, 5.0],
            [0.0, 5.0],
            [-2.0, -3.0],
        ],
        dtype=torch.float64,
    )

    clearance = world.boundary_clearance(positions)

    assert torch.allclose(
        clearance,
        torch.tensor([5.0, 0.0, -3.0], dtype=torch.float64),
    )


def test_obstacle_clearance_delegates_to_circle_geometry():
    obstacles = make_obstacles([[5.0, 5.0]], [1.0])
    world = WorldState(10.0, 10.0, obstacles)

    positions = torch.tensor(
        [
            [7.0, 5.0],
            [6.0, 5.0],
            [5.5, 5.0],
        ]
    )

    clearance = world.obstacle_clearance(positions)

    assert clearance.shape == (3, 1)
    assert torch.allclose(
        clearance[:, 0],
        torch.tensor([1.0, 0.0, -0.5]),
    )


def test_from_config_accepts_matching_explicit_layout():
    config = WorldConfig(
        width_m=20.0,
        height_m=30.0,
        obstacle_count=1,
        obstacle_radius_min_m=1.0,
        obstacle_radius_max_m=2.0,
    )
    obstacles = make_obstacles([[5.0, 5.0]], [1.5])

    world = WorldState.from_config(config, obstacles)

    assert world.width_m == 20.0
    assert world.height_m == 30.0
    assert world.obstacle_count == 1


def test_from_config_rejects_obstacle_count_mismatch():
    config = WorldConfig(obstacle_count=2)
    obstacles = make_obstacles([[5.0, 5.0]], [1.0])

    with pytest.raises(ValueError, match="obstacle count"):
        WorldState.from_config(config, obstacles)


def test_from_config_supports_zero_obstacle_configuration():
    config = WorldConfig(obstacle_count=0)
    obstacles = CircleObstacles.empty()

    world = WorldState.from_config(config, obstacles)

    assert world.obstacle_count == 0


def test_from_config_rejects_invalid_world_config():
    config = WorldConfig(width_m=-1.0, obstacle_count=0)
    obstacles = CircleObstacles.empty()

    with pytest.raises(ValueError):
        WorldState.from_config(config, obstacles)


@pytest.mark.parametrize(
    "width,height",
    [
        (0.0, 10.0),
        (10.0, 0.0),
        (-1.0, 10.0),
        (10.0, -1.0),
        (float("inf"), 10.0),
        (10.0, float("nan")),
        (True, 10.0),
    ],
)
def test_invalid_world_dimensions_are_rejected(width, height):
    with pytest.raises((TypeError, ValueError)):
        WorldState(
            width_m=width,
            height_m=height,
            obstacles=CircleObstacles.empty(),
        )


def test_world_state_rejects_wrong_obstacle_type():
    with pytest.raises(TypeError, match="CircleObstacles"):
        WorldState(
            width_m=10.0,
            height_m=10.0,
            obstacles=None,
        )


def test_world_state_preserves_float64_dtype():
    obstacles = make_obstacles(
        [[5.0, 5.0]],
        [1.0],
        dtype=torch.float64,
    )
    world = WorldState(10.0, 10.0, obstacles)
    positions = torch.tensor([[7.0, 5.0]], dtype=torch.float64)

    boundary = world.boundary_clearance(positions)
    obstacle = world.obstacle_clearance(positions)

    assert boundary.dtype == torch.float64
    assert obstacle.dtype == torch.float64


def test_world_state_preserves_cuda_device_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda")
    obstacles = make_obstacles(
        [[5.0, 5.0]],
        [1.0],
        device=device,
    )
    world = WorldState(10.0, 10.0, obstacles)
    positions = torch.tensor(
        [[5.0, 5.0], [0.0, 0.0]],
        device=device,
    )

    inside = world.contains_positions(positions)
    boundary = world.boundary_clearance(positions)
    obstacle = world.obstacle_clearance(positions)

    assert inside.device == positions.device
    assert boundary.device == positions.device
    assert obstacle.device == positions.device
    assert inside.tolist() == [True, True]
