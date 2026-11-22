import dataclasses
import math

import pytest
import torch

from lawless.world.config import WorldConfig
from lawless.world.geometry import CircleObstacles
from lawless.world.generated_world import GeneratedWorld
from lawless.world.world_state import WorldState


def make_world(dtype=torch.float32, device="cpu"):
    config = WorldConfig(
        width_m=20.0,
        height_m=20.0,
        obstacle_count=1,
        obstacle_radius_min_m=1.0,
        obstacle_radius_max_m=1.0,
    )

    obstacles = CircleObstacles(
        centers=torch.tensor(
            [[3.0, 3.0]],
            dtype=dtype,
            device=device,
        ),
        radii=torch.tensor(
            [1.0],
            dtype=dtype,
            device=device,
        ),
    )
    return WorldState.from_config(config, obstacles)


def make_generated_world(dtype=torch.float32, device="cpu"):
    world = make_world(dtype=dtype, device=device)

    return GeneratedWorld(
        world=world,
        spawn_positions=torch.tensor(
            [[10.0, 10.0]],
            dtype=dtype,
            device=device,
        ),
        agent_body_radius_m=0.5,
        extraction_center=torch.tensor(
            [17.0, 17.0],
            dtype=dtype,
            device=device,
        ),
        extraction_radius_m=1.0,
        core_positions=torch.tensor(
            [[13.0, 10.0], [10.0, 14.0]],
            dtype=dtype,
            device=device,
        ),
        core_radii=torch.tensor(
            [0.5, 0.5],
            dtype=dtype,
            device=device,
        ),
        seed=42,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_generated_world_accepts_valid_complete_layout(dtype):
    generated = make_generated_world(dtype=dtype)

    assert isinstance(generated.world, WorldState)
    assert generated.spawn_positions.shape == (1, 2)
    assert generated.extraction_center.shape == (2,)
    assert generated.core_positions.shape == (2, 2)
    assert generated.core_radii.shape == (2,)

    assert generated.agent_body_radius_m == float(
        torch.tensor(0.5, dtype=dtype).item()
    )
    assert generated.extraction_radius_m == float(
        torch.tensor(1.0, dtype=dtype).item()
    )
    assert generated.seed == 42

    tensors = (
        generated.world.obstacles.centers,
        generated.world.obstacles.radii,
        generated.spawn_positions,
        generated.extraction_center,
        generated.core_positions,
        generated.core_radii,
    )
    assert all(t.dtype == dtype for t in tensors)
    assert all(t.device.type == "cpu" for t in tensors)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_radii_are_rounded_to_layout_dtype(dtype):
    configured_agent_radius = 0.1
    configured_extraction_radius = 0.3

    generated = dataclasses.replace(
        make_generated_world(dtype=dtype),
        agent_body_radius_m=configured_agent_radius,
        extraction_radius_m=configured_extraction_radius,
    )

    expected_agent = float(
        torch.tensor(configured_agent_radius, dtype=dtype).item()
    )
    expected_extraction = float(
        torch.tensor(configured_extraction_radius, dtype=dtype).item()
    )

    assert generated.agent_body_radius_m == expected_agent
    assert generated.extraction_radius_m == expected_extraction


def test_generated_world_is_structurally_frozen():
    generated = make_generated_world()

    with pytest.raises(dataclasses.FrozenInstanceError):
        generated.seed = 7

    # Tensor storage itself remains mutable in PyTorch. This test documents
    # that the frozen dataclass is structural, not a deep immutable container.
    original_x = generated.spawn_positions[0, 0].item()
    generated.spawn_positions[0, 0] = original_x + 0.25
    assert generated.spawn_positions[0, 0].item() == original_x + 0.25


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("agent_body_radius_m", 0.0, ValueError),
        ("agent_body_radius_m", -0.1, ValueError),
        ("agent_body_radius_m", float("nan"), ValueError),
        ("agent_body_radius_m", float("inf"), ValueError),
        ("agent_body_radius_m", True, TypeError),
        ("extraction_radius_m", 0.0, ValueError),
        ("extraction_radius_m", -1.0, ValueError),
        ("extraction_radius_m", float("nan"), ValueError),
        ("extraction_radius_m", float("inf"), ValueError),
        ("extraction_radius_m", True, TypeError),
    ],
)
def test_invalid_scalar_radii_are_rejected(field, value, error):
    generated = make_generated_world()

    with pytest.raises(error):
        dataclasses.replace(generated, **{field: value})






@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("spawn_positions", torch.zeros((2,)), ValueError),
        ("spawn_positions", torch.zeros((1, 3)), ValueError),
        ("extraction_center", torch.zeros((1, 2)), ValueError),
        ("core_positions", torch.zeros((2,)), ValueError),
        ("core_radii", torch.zeros((2, 1)), ValueError),
        ("core_radii", torch.zeros((1,)), ValueError),
    ],
)
def test_invalid_tensor_shapes_are_rejected(field, value, error):
    generated = make_generated_world()

    with pytest.raises(error):
        dataclasses.replace(generated, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spawn_positions", torch.tensor([[float("nan"), 10.0]])),
        ("extraction_center", torch.tensor([17.0, float("inf")])),
        ("core_positions", torch.tensor([[float("nan"), 10.0], [10.0, 14.0]])),
        ("core_radii", torch.tensor([0.5, float("inf")])),
    ],
)
def test_nonfinite_layout_tensors_are_rejected(field, value):
    generated = make_generated_world()

    with pytest.raises(ValueError):
        dataclasses.replace(generated, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spawn_positions", torch.tensor([[10.0, 10.0]], dtype=torch.float64)),
        ("extraction_center", torch.tensor([17.0, 17.0], dtype=torch.float64)),
        ("core_positions", torch.tensor([[13.0, 10.0], [10.0, 14.0]], dtype=torch.float64)),
        ("core_radii", torch.tensor([0.5, 0.5], dtype=torch.float64)),
    ],
)
def test_layout_tensor_dtype_must_match_world(field, value):
    generated = make_generated_world(dtype=torch.float32)

    with pytest.raises(ValueError, match="dtype"):
        dataclasses.replace(generated, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("spawn_positions", torch.tensor([[10.0, 10.0]], device="meta")),
        ("extraction_center", torch.tensor([17.0, 17.0], device="meta")),
        ("core_positions", torch.empty((2, 2), device="meta")),
        ("core_radii", torch.empty((2,), device="meta")),
    ],
)
def test_layout_tensor_device_must_match_world(field, value):
    generated = make_generated_world()

    with pytest.raises(ValueError, match="device"):
        dataclasses.replace(generated, **{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("seed", -1),
        ("seed", 1 << 63),
        ("seed", 1.5),
    ],
)
def test_invalid_seed_provenance_is_rejected(field, value):
    generated = make_generated_world()

    error = TypeError if isinstance(value, float) or isinstance(value, bool) else ValueError
    with pytest.raises(error):
        dataclasses.replace(generated, **{field: value})


@pytest.mark.parametrize(
    ("spawn_positions", "message"),
    [
        (torch.tensor([[0.25, 10.0]]), "spawn body circle"),
        (torch.tensor([[3.0, 3.0]]), "spawn-obstacle"),
        (torch.tensor([[10.0, 10.0], [10.5, 10.0]]), "spawn-spawn"),
        (torch.tensor([[16.0, 17.0]]), "spawn-extraction"),
    ],
)
def test_spawn_body_containment_and_clearance_are_strict(
    spawn_positions,
    message,
):
    generated = make_generated_world()

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(generated, spawn_positions=spawn_positions)


@pytest.mark.parametrize(
    ("extraction_center", "message"),
    [
        (torch.tensor([0.5, 10.0]), "extraction circle"),
        (torch.tensor([3.0, 3.0]), "extraction-obstacle"),
    ],
)
def test_extraction_containment_and_obstacle_clearance_are_strict(
    extraction_center,
    message,
):
    generated = make_generated_world()

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(
            generated,
            extraction_center=extraction_center,
        )


@pytest.mark.parametrize(
    ("core_positions", "message"),
    [
        (torch.tensor([[0.25, 10.0], [10.0, 14.0]]), "core circle"),
        (torch.tensor([[3.0, 3.0], [10.0, 14.0]]), "core-obstacle"),
        (torch.tensor([[10.0, 10.0], [10.0, 14.0]]), "core-spawn"),
        (torch.tensor([[17.0, 17.0], [10.0, 14.0]]), "core-extraction"),
        (torch.tensor([[13.0, 10.0], [14.0, 10.0]]), "core-core"),
    ],
)
def test_core_containment_and_clearance_are_strict(core_positions, message):
    generated = make_generated_world()

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(generated, core_positions=core_positions)


def test_zero_spawn_and_zero_core_layouts_are_supported():
    generated = make_generated_world()

    empty = dataclasses.replace(
        generated,
        spawn_positions=torch.empty((0, 2), dtype=torch.float32),
        core_positions=torch.empty((0, 2), dtype=torch.float32),
        core_radii=torch.empty((0,), dtype=torch.float32),
    )

    assert empty.spawn_positions.shape == (0, 2)
    assert empty.core_positions.shape == (0, 2)
    assert empty.core_radii.shape == (0,)


def test_existing_world_state_constructor_and_delegation_remain_unchanged():
    world = make_world()

    assert world.width_m == 20.0
    assert world.height_m == 20.0
    assert world.obstacle_count == 1

    points = torch.tensor([[3.0, 3.0], [10.0, 10.0]])
    clearance = world.obstacle_clearance(points)

    assert clearance.shape == (2, 1)
    assert clearance[0, 0] < 0
    assert clearance[1, 0] > 0


def test_scalar_radius_python_zero_is_rejected():
    generated = make_generated_world(dtype=torch.float64)

    # 1e-400 is already 0.0 as a Python float; no tensor conversion is involved.
    with pytest.raises(ValueError, match="finite and positive"):
        dataclasses.replace(generated, agent_body_radius_m=1e-400)


def test_scalar_radius_python_infinity_is_rejected():
    generated = make_generated_world(dtype=torch.float64)

    # 1e309 is already inf as a Python float; no tensor conversion is involved.
    with pytest.raises(ValueError, match="finite and positive"):
        dataclasses.replace(generated, extraction_radius_m=1e309)


def test_scalar_radius_finite_python_value_underflowing_to_float32_zero_is_rejected():
    generated = make_generated_world(dtype=torch.float32)
    radius = 1e-50

    # Establish the input is valid as a Python float, but becomes zero in float32.
    assert math.isfinite(radius)
    assert radius > 0.0
    assert torch.tensor(radius, dtype=torch.float32).item() == 0.0

    with pytest.raises(ValueError, match="layout dtype"):
        dataclasses.replace(generated, agent_body_radius_m=radius)


def test_scalar_radius_finite_python_value_overflowing_to_float32_infinity_is_rejected():
    generated = make_generated_world(dtype=torch.float32)
    radius = 1e39

    # Establish the input is valid as a Python float, but becomes infinity in float32.
    assert math.isfinite(radius)
    assert radius > 0.0
    assert math.isinf(torch.tensor(radius, dtype=torch.float32).item())

    with pytest.raises(ValueError, match="layout dtype"):
        dataclasses.replace(generated, extraction_radius_m=radius)


@pytest.mark.parametrize(
    ("dtype", "radius"),
    [
        (torch.float32, 1e-40),
        (torch.float64, 1e-320),
    ],
)
def test_scalar_radius_positive_representable_subnormal_is_accepted(dtype, radius):
    generated = make_generated_world(dtype=dtype)

    # Both values are finite, positive, and remain positive in their target dtype.
    assert math.isfinite(radius)
    assert radius > 0.0

    converted_radius = torch.tensor(radius, dtype=dtype).item()
    assert converted_radius > 0.0

    result = dataclasses.replace(generated, agent_body_radius_m=radius)

    # GeneratedWorld normalizes the scalar to the target dtype's representable value.
    assert result.agent_body_radius_m == converted_radius
    assert result.agent_body_radius_m > 0.0

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_generated_world_valid_cuda_construction_matches_cpu(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    cpu = make_generated_world(dtype=dtype, device="cpu")
    cuda = make_generated_world(dtype=dtype, device="cuda")

    cpu_tensors = (
        cpu.world.obstacles.centers,
        cpu.world.obstacles.radii,
        cpu.spawn_positions,
        cpu.extraction_center,
        cpu.core_positions,
        cpu.core_radii,
    )
    cuda_tensors = (
        cuda.world.obstacles.centers,
        cuda.world.obstacles.radii,
        cuda.spawn_positions,
        cuda.extraction_center,
        cuda.core_positions,
        cuda.core_radii,
    )

    assert all(t.device.type == "cpu" for t in cpu_tensors)
    assert all(t.device.type == "cuda" for t in cuda_tensors)
    assert all(t.dtype == dtype for t in cuda_tensors)

    for cpu_tensor, cuda_tensor in zip(cpu_tensors, cuda_tensors):
        torch.testing.assert_close(
            cpu_tensor,
            cuda_tensor.cpu(),
            rtol=0,
            atol=0,
        )

    expected_agent_radius = float(
        torch.tensor(0.5, dtype=dtype).item()
    )
    expected_extraction_radius = float(
        torch.tensor(1.0, dtype=dtype).item()
    )

    assert cuda.agent_body_radius_m == expected_agent_radius
    assert cuda.extraction_radius_m == expected_extraction_radius
    assert cuda.seed == cpu.seed


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_generated_world_cuda_rejects_spawn_obstacle_tangency(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    generated = make_generated_world(dtype=dtype, device="cuda")

    # Obstacle center is (3, 3), obstacle radius is 1, and agent body
    # radius is 0.5. This spawn center is exactly 1.5 m away from the
    # obstacle center, so the circles are tangent and must be rejected.
    tangent_spawn = torch.tensor(
        [[4.5, 3.0]],
        dtype=dtype,
        device="cuda",
    )

    assert tangent_spawn.device.type == "cuda"

    with pytest.raises(ValueError, match="spawn-obstacle"):
        dataclasses.replace(
            generated,
            spawn_positions=tangent_spawn,
        )
