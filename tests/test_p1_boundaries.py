import pytest
import torch

from lawless.world.boundaries import (
    boundary_clearance,
    inside_world,
    outside_world,
)


def test_default_world_extent_is_100_by_100_metres():
    points = torch.tensor(
        [
            [0.0, 0.0],
            [100.0, 100.0],
            [50.0, 50.0],
            [100.01, 50.0],
            [50.0, -0.01],
        ],
        dtype=torch.float64,
    )

    assert inside_world(points).tolist() == [
        True, True, True, False, False
    ]
    assert outside_world(points).tolist() == [
        False, False, False, True, True
    ]


def test_all_four_edges_are_inside():
    points = torch.tensor(
        [
            [0.0, 50.0],
            [100.0, 50.0],
            [50.0, 0.0],
            [50.0, 100.0],
        ]
    )

    assert inside_world(points).tolist() == [True, True, True, True]


def test_all_four_corners_are_inside():
    corners = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 100.0],
            [100.0, 0.0],
            [100.0, 100.0],
        ]
    )

    assert inside_world(corners).all().item()


def test_boundary_classification_uses_strict_outside_comparison():
    epsilon = 1e-4
    points = torch.tensor(
        [
            [-epsilon, 50.0],
            [0.0, 50.0],
            [epsilon, 50.0],
            [100.0 - epsilon, 50.0],
            [100.0, 50.0],
            [100.0 + epsilon, 50.0],
        ]
    )

    assert inside_world(points).tolist() == [
        False, True, True, True, True, False
    ]


def test_custom_world_dimensions():
    points = torch.tensor(
        [
            [20.0, 10.0],
            [20.01, 10.0],
            [10.0, 10.0],
        ]
    )

    assert inside_world(
        points, width_m=20.0, height_m=10.0
    ).tolist() == [True, False, True]


def test_boundary_clearance_inside_edges_and_outside():
    points = torch.tensor(
        [
            [50.0, 50.0],
            [0.0, 50.0],
            [100.0, 50.0],
            [-2.0, 50.0],
            [50.0, 103.0],
        ],
        dtype=torch.float64,
    )

    clearance = boundary_clearance(points)

    assert clearance.shape == (5,)
    assert clearance.dtype == torch.float64
    assert torch.allclose(
        clearance,
        torch.tensor([50.0, 0.0, 0.0, -2.0, -3.0], dtype=torch.float64),
    )


def test_boundary_functions_support_arbitrary_leading_dimensions():
    points = torch.tensor(
        [
            [[1.0, 1.0], [50.0, 50.0]],
            [[100.0, 100.0], [-1.0, 50.0]],
        ],
        dtype=torch.float32,
    )

    inside = inside_world(points)
    clearance = boundary_clearance(points)

    assert inside.shape == (2, 2)
    assert clearance.shape == (2, 2)
    assert inside.dtype == torch.bool
    assert clearance.dtype == points.dtype
    assert inside.device == points.device
    assert clearance.device == points.device


def test_empty_position_batch_is_supported():
    points = torch.empty((0, 2), dtype=torch.float32)

    assert inside_world(points).shape == (0,)
    assert outside_world(points).shape == (0,)
    assert boundary_clearance(points).shape == (0,)


@pytest.mark.parametrize(
    "width,height",
    [
        (0.0, 100.0),
        (100.0, 0.0),
        (-1.0, 100.0),
        (100.0, -1.0),
        (float("inf"), 100.0),
        (100.0, float("nan")),
    ],
)
def test_invalid_world_dimensions_are_rejected(width, height):
    points = torch.zeros((1, 2))

    with pytest.raises(ValueError):
        inside_world(points, width_m=width, height_m=height)


def test_boolean_world_dimension_is_rejected():
    with pytest.raises(TypeError):
        inside_world(torch.zeros((1, 2)), width_m=True)


@pytest.mark.parametrize(
    "points",
    [
        torch.zeros((2, 3)),
        torch.zeros((2, 2), dtype=torch.int64),
        torch.tensor([[0.0, float("nan")]]),
        torch.tensor([[0.0, float("inf")]]),
    ],
)
def test_invalid_positions_are_rejected(points):
    with pytest.raises((TypeError, ValueError)):
        inside_world(points)


def test_boundary_functions_preserve_cuda_device_when_available():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    points = torch.tensor(
        [[0.0, 0.0], [50.0, 50.0], [101.0, 50.0]],
        device="cuda",
    )

    inside = inside_world(points)
    clearance = boundary_clearance(points)

    assert inside.device.type == "cuda"
    assert clearance.device.type == "cuda"
    assert inside.tolist() == [True, True, False]
