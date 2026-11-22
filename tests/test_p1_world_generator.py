import pytest
import torch

from lawless.world.config import WorldConfig
from lawless.world.generator import (
    WorldGenerationError,
    generate_obstacle_world,
)


def make_config(
    *,
    width=100.0,
    height=100.0,
    count=20,
    radius_min=1.0,
    radius_max=3.0,
):
    return WorldConfig(
        width_m=width,
        height_m=height,
        obstacle_count=count,
        obstacle_radius_min_m=radius_min,
        obstacle_radius_max_m=radius_max,
    )


def assert_obstacle_invariants(world):
    centers = world.obstacles.centers
    radii = world.obstacles.radii

    assert centers.ndim == 2
    assert centers.shape[1] == 2
    assert radii.shape == (centers.shape[0],)
    assert centers.dtype == radii.dtype
    assert centers.device == radii.device
    assert torch.isfinite(centers).all()
    assert torch.isfinite(radii).all()
    assert (radii > 0).all()

    if radii.numel() == 0:
        return

    x, y = centers.unbind(dim=-1)

    assert ((x - radii) >= 0).all()
    assert ((x + radii) <= world.width_m).all()
    assert ((y - radii) >= 0).all()
    assert ((y + radii) <= world.height_m).all()

    if centers.shape[0] > 1:
        delta = centers[:, None, :] - centers[None, :, :]
        distances = torch.linalg.vector_norm(delta, dim=-1)
        summed_radii = radii[:, None] + radii[None, :]
        upper = torch.triu(
            torch.ones_like(distances, dtype=torch.bool),
            diagonal=1,
        )

        # Strict positive clearance: tangency is not accepted.
        assert (distances[upper] > summed_radii[upper]).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_seeded_obstacles_are_deterministic(dtype):
    config = make_config()

    first = generate_obstacle_world(
        config,
        seed=42,
        dtype=dtype,
    )
    second = generate_obstacle_world(
        config,
        seed=42,
        dtype=dtype,
    )

    torch.testing.assert_close(
        first.obstacles.centers,
        second.obstacles.centers,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.obstacles.radii,
        second.obstacles.radii,
        rtol=0,
        atol=0,
    )

    assert_obstacle_invariants(first)


def test_generation_does_not_change_global_rng_state():
    config = make_config()

    torch.manual_seed(9876)
    state_before = torch.random.get_rng_state().clone()

    generate_obstacle_world(config, seed=123)

    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)


def test_generation_is_independent_of_global_rng_history():
    config = make_config()

    torch.manual_seed(1)
    first = generate_obstacle_world(config, seed=2026)

    torch.manual_seed(999999)
    _ = torch.rand(1000)
    second = generate_obstacle_world(config, seed=2026)

    torch.testing.assert_close(
        first.obstacles.centers,
        second.obstacles.centers,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        first.obstacles.radii,
        second.obstacles.radii,
        rtol=0,
        atol=0,
    )


def test_different_seeds_change_layout():
    config = make_config()

    first = generate_obstacle_world(config, seed=11)
    second = generate_obstacle_world(config, seed=12)

    assert not torch.equal(
        first.obstacles.centers,
        second.obstacles.centers,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_obstacles_is_supported(dtype):
    config = make_config(count=0)

    world = generate_obstacle_world(
        config,
        seed=7,
        dtype=dtype,
    )

    assert world.obstacle_count == 0
    assert world.obstacles.centers.shape == (0, 2)
    assert world.obstacles.radii.shape == (0,)
    assert world.obstacles.centers.dtype == dtype
    assert world.obstacles.radii.dtype == dtype

    assert_obstacle_invariants(world)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_obstacles_allow_small_world_with_unused_large_radii(dtype):
    config = make_config(
        width=1.0,
        height=0.75,
        count=0,
        radius_min=2.0,
        radius_max=3.0,
    )

    world = generate_obstacle_world(
        config,
        seed=7,
        dtype=dtype,
    )

    assert world.obstacle_count == 0
    assert world.obstacles.centers.shape == (0, 2)
    assert world.obstacles.radii.shape == (0,)
    assert_obstacle_invariants(world)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fixed_decimal_radius_uses_dtype_rounded_bound(dtype):
    configured_radius = 0.1
    config = make_config(
        count=8,
        radius_min=configured_radius,
        radius_max=configured_radius,
    )

    world = generate_obstacle_world(
        config,
        seed=42,
        dtype=dtype,
    )

    expected_radius = torch.tensor(configured_radius, dtype=dtype)

    assert torch.isfinite(world.obstacles.radii).all()
    assert (world.obstacles.radii > 0).all()
    assert (world.obstacles.radii == expected_radius).all()

    # The configured decimal is interpreted as its nearest representable
    # value in the selected output dtype, not as exact real-number equality.
    assert_obstacle_invariants(world)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sampled_radii_stay_within_dtype_rounded_configured_bounds(dtype):
    radius_min = 0.1
    radius_max = 0.3

    config = make_config(
        count=20,
        radius_min=radius_min,
        radius_max=radius_max,
    )

    world = generate_obstacle_world(
        config,
        seed=314,
        dtype=dtype,
    )

    lower = torch.tensor(radius_min, dtype=dtype)
    upper = torch.tensor(radius_max, dtype=dtype)

    assert (world.obstacles.radii >= lower).all()
    assert (world.obstacles.radii <= upper).all()
    assert_obstacle_invariants(world)


@pytest.mark.parametrize(
    ("radius_min", "radius_max", "dtype"),
    [
        # float32 lower bound rounds to zero: positive radius is lost.
        (1e-50, 1e-40, torch.float32),
        # float32 upper bound overflows to infinity.
        (1.0, 1e39, torch.float32),
        # Both float32 bounds overflow.
        (1e39, 1e40, torch.float32),
        # float64 lower bound underflows to zero.
        (1e-400, 1e-300, torch.float64),
        # float64 upper bound overflows to infinity.
        (1.0, 1e309, torch.float64),
    ],
)
def test_radius_bounds_not_representable_as_finite_positive_dtype_values_are_rejected(
    radius_min,
    radius_max,
    dtype,
):
    config = make_config(
        count=1,
        radius_min=radius_min,
        radius_max=radius_max,
    )

    with pytest.raises(ValueError):
        generate_obstacle_world(
            config,
            seed=3,
            dtype=dtype,
        )


@pytest.mark.parametrize(
    ("radius_min", "radius_max", "dtype"),
    [
        # Distinct Python bounds collapse to the same float32 value.
        (1.0, 1.0 + 1e-8, torch.float32),
        # This float64 pair is already equal in Python before conversion.
        (1.0, 1.0 + 1e-17, torch.float64),
    ],
)
def test_radius_bounds_that_collapse_in_output_dtype_are_valid_fixed_radii(
    radius_min,
    radius_max,
    dtype,
):
    config = make_config(
        count=8,
        radius_min=radius_min,
        radius_max=radius_max,
    )

    first = generate_obstacle_world(
        config,
        seed=3,
        dtype=dtype,
    )
    second = generate_obstacle_world(
        config,
        seed=3,
        dtype=dtype,
    )

    expected_radius = torch.tensor(radius_min, dtype=dtype)

    assert torch.equal(
        torch.tensor(radius_min, dtype=dtype),
        torch.tensor(radius_max, dtype=dtype),
    )
    assert (first.obstacles.radii == expected_radius).all()
    assert torch.equal(
        first.obstacles.centers,
        second.obstacles.centers,
    )
    assert torch.equal(
        first.obstacles.radii,
        second.obstacles.radii,
    )
    assert_obstacle_invariants(first)


@pytest.mark.parametrize(
    ("seed", "attempts", "dtype", "error"),
    [
        (True, 10, torch.float32, TypeError),
        (-1, 10, torch.float32, ValueError),
        (1.5, 10, torch.float32, TypeError),
        (1, True, torch.float32, TypeError),
        (1, 0, torch.float32, ValueError),
        (1, 1.5, torch.float32, TypeError),
        (1, 10, torch.float16, TypeError),
        (1, 10, torch.bfloat16, TypeError),
    ],
)
def test_strict_seed_attempt_limit_and_dtype_validation(
    seed,
    attempts,
    dtype,
    error,
):
    with pytest.raises(error):
        generate_obstacle_world(
            make_config(),
            seed=seed,
            max_attempts_per_obstacle=attempts,
            dtype=dtype,
        )


def test_seed_above_application_policy_maximum_is_rejected():
    with pytest.raises(ValueError):
        generate_obstacle_world(
            make_config(),
            seed=(1 << 63),
        )


def test_non_world_config_duck_type_is_rejected():
    class DuckConfig:
        width_m = 100.0
        height_m = 100.0
        obstacle_count = 0
        obstacle_radius_min_m = 1.0
        obstacle_radius_max_m = 2.0

    with pytest.raises(TypeError, match="WorldConfig"):
        generate_obstacle_world(DuckConfig(), seed=1)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (make_config(width=float("nan")), ValueError),
        (make_config(height=float("inf")), ValueError),
        (make_config(count=-1), ValueError),
        # Canonical WorldConfig validation raises ValueError for this case.
        (make_config(count=1.5), ValueError),
        (make_config(radius_min=0.0), ValueError),
        (make_config(radius_min=4.0, radius_max=3.0), ValueError),
        (
            make_config(
                width=5.0,
                height=5.0,
                radius_min=3.0,
                radius_max=3.0,
            ),
            ValueError,
        ),
    ],
)
def test_canonical_world_config_validation_is_preserved(config, error):
    with pytest.raises(error):
        generate_obstacle_world(config, seed=1)


def test_impossible_packing_fails_deterministically_without_partial_world():
    # A radius-1 circle in a 2x2 world has exactly one possible center.
    # Two such circles cannot have strictly positive mutual clearance.
    config = make_config(
        width=2.0,
        height=2.0,
        count=2,
        radius_min=1.0,
        radius_max=1.0,
    )

    messages = []

    for _ in range(2):
        with pytest.raises(WorldGenerationError) as exc:
            generate_obstacle_world(
                config,
                seed=55,
                max_attempts_per_obstacle=4,
            )
        messages.append(str(exc.value))

    assert messages[0] == messages[1]
    assert "category='obstacle'" in messages[0]
    assert "index=1" in messages[0]
    assert "max_attempts=4" in messages[0]
    assert "seed=55" in messages[0]
    assert "No partial world was returned" in messages[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cuda_layouts_are_exact_after_transfer(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    config = make_config()

    cpu_world = generate_obstacle_world(
        config,
        seed=314159,
        device="cpu",
        dtype=dtype,
    )
    cuda_world = generate_obstacle_world(
        config,
        seed=314159,
        device="cuda",
        dtype=dtype,
    )

    assert cuda_world.obstacles.centers.device.type == "cuda"
    assert cuda_world.obstacles.radii.device.type == "cuda"
    assert cuda_world.obstacles.centers.dtype == dtype
    assert cuda_world.obstacles.radii.dtype == dtype

    torch.testing.assert_close(
        cpu_world.obstacles.centers,
        cuda_world.obstacles.centers.cpu(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        cpu_world.obstacles.radii,
        cuda_world.obstacles.radii.cpu(),
        rtol=0,
        atol=0,
    )

    assert_obstacle_invariants(cpu_world)
    assert_obstacle_invariants(cuda_world)
