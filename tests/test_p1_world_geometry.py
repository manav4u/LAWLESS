import pytest
import torch

from lawless.world.geometry import (
    CircleObstacles,
    circle_circle_contact,
    circle_circle_signed_clearance,
    point_circle_contact,
    point_circle_signed_clearance,
)


def test_circle_obstacle_representation_and_units():
    obstacles = CircleObstacles(
        centers=torch.tensor([[3.0, 4.0]], dtype=torch.float64),
        radii=torch.tensor([2.0], dtype=torch.float64),
    )

    assert obstacles.count == 1
    assert obstacles.centers.shape == (1, 2)
    assert obstacles.radii.shape == (1,)
    assert obstacles.centers.dtype == torch.float64
    assert obstacles.radii.dtype == torch.float64


def test_point_circle_clearance_outside_tangent_and_inside():
    obstacles = CircleObstacles(
        centers=torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        radii=torch.tensor([2.0], dtype=torch.float64),
    )
    points = torch.tensor(
        [[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
        dtype=torch.float64,
    )

    clearance = point_circle_signed_clearance(points, obstacles)

    assert clearance.shape == (3, 1)
    assert torch.allclose(
        clearance[:, 0],
        torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64),
    )
    assert point_circle_contact(points, obstacles).tolist() == [
        [False],
        [True],
        [True],
    ]


def test_point_circle_batched_leading_dimensions():
    obstacles = CircleObstacles(
        centers=torch.tensor([[0.0, 0.0], [5.0, 0.0]]),
        radii=torch.tensor([1.0, 2.0]),
    )
    points = torch.zeros((2, 3, 2), dtype=torch.float32)

    result = point_circle_signed_clearance(points, obstacles)

    assert result.shape == (2, 3, 2)
    assert result.dtype == points.dtype
    assert result.device == points.device
    assert torch.allclose(
        result,
        torch.tensor([[-1.0, 3.0]]).expand(2, 3, 2),
    )


def test_zero_obstacles_are_valid_and_preserve_shape():
    obstacles = CircleObstacles.empty(dtype=torch.float64)
    points = torch.zeros((4, 3, 2), dtype=torch.float64)

    clearance = point_circle_signed_clearance(points, obstacles)
    contacts = point_circle_contact(points, obstacles)

    assert obstacles.count == 0
    assert clearance.shape == (4, 3, 0)
    assert contacts.shape == (4, 3, 0)
    assert clearance.dtype == torch.float64
    assert contacts.dtype == torch.bool


def test_empty_obstacles_can_be_constructed_directly():
    obstacles = CircleObstacles(
        centers=torch.empty((0, 2), dtype=torch.float32),
        radii=torch.empty((0,), dtype=torch.float32),
    )
    assert obstacles.count == 0


def test_circle_circle_clearance_and_inclusive_tangency():
    centers_a = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
    radii_a = torch.tensor([1.0, 1.0])
    centers_b = torch.tensor([[2.0, 0.0], [10.5, 0.0]])
    radii_b = torch.tensor([1.0, 1.0])

    clearance = circle_circle_signed_clearance(
        centers_a, radii_a, centers_b, radii_b
    )
    contacts = circle_circle_contact(
        centers_a, radii_a, centers_b, radii_b
    )

    assert clearance.shape == (2, 2)
    assert torch.allclose(
        clearance,
        torch.tensor([[0.0, 8.5], [6.0, -1.5]]),
    )
    assert contacts.tolist() == [
        [True, False],
        [False, True],
    ]


def test_circle_circle_empty_pair_dimension():
    centers_a = torch.empty((0, 2), dtype=torch.float32)
    radii_a = torch.empty((0,), dtype=torch.float32)
    centers_b = torch.tensor([[1.0, 1.0]])
    radii_b = torch.tensor([1.0])

    result = circle_circle_contact(
        centers_a, radii_a, centers_b, radii_b
    )
    assert result.shape == (0, 1)
    assert result.dtype == torch.bool


@pytest.mark.parametrize(
    "centers,radii",
    [
        (torch.zeros((2, 3)), torch.ones(2)),
        (torch.zeros((2, 2)), torch.ones(1)),
        (torch.zeros((2, 2)), torch.ones((2, 1))),
        (torch.zeros((2, 2)), torch.tensor([1.0, 0.0])),
        (torch.zeros((2, 2)), torch.tensor([1.0, -1.0])),
        (torch.tensor([[0.0, float("nan")]]), torch.ones(1)),
        (torch.zeros((2, 2), dtype=torch.float32),
         torch.ones(2, dtype=torch.float64)),
    ],
)
def test_invalid_obstacle_data_is_rejected(centers, radii):
    with pytest.raises((TypeError, ValueError)):
        CircleObstacles(centers=centers, radii=radii)


def test_point_and_obstacle_dtype_mismatch_is_rejected():
    obstacles = CircleObstacles(
        centers=torch.zeros((1, 2), dtype=torch.float32),
        radii=torch.ones(1, dtype=torch.float32),
    )
    points = torch.zeros((2, 2), dtype=torch.float64)

    with pytest.raises(ValueError, match="same dtype"):
        point_circle_signed_clearance(points, obstacles)


def test_point_and_obstacle_device_mismatch_is_rejected_when_cuda_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    obstacles = CircleObstacles(
        centers=torch.zeros((1, 2), device="cuda"),
        radii=torch.ones(1, device="cuda"),
    )
    points = torch.zeros((1, 2), device="cpu")

    with pytest.raises(ValueError, match="same device"):
        point_circle_contact(points, obstacles)


def test_geometry_preserves_cuda_device_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda")
    obstacles = CircleObstacles(
        centers=torch.tensor([[0.0, 0.0]], device=device),
        radii=torch.tensor([1.0], device=device),
    )
    points = torch.tensor([[1.0, 0.0], [2.0, 0.0]], device=device)

    clearance = point_circle_signed_clearance(points, obstacles)
    contacts = point_circle_contact(points, obstacles)

    assert clearance.device == device
    assert contacts.device == device
    assert contacts.tolist() == [[True], [False]]
