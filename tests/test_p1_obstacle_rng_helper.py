import pytest
import torch

from lawless.world.config import WorldConfig
from lawless.world.generator import (
    _generate_obstacles_with_rng,
    _validate_inputs,
    generate_obstacle_world,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_obstacle_rng_helper_consumes_supplied_cpu_generator(dtype):
    config = WorldConfig()
    seed = 7319
    max_attempts = 10_000

    width, height, count, radius_min, radius_max = _validate_inputs(
        config,
        seed,
        max_attempts,
        dtype,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    state_before = generator.get_state().clone()

    obstacles = _generate_obstacles_with_rng(
        width=width,
        height=height,
        count=count,
        radius_min=radius_min,
        radius_max=radius_max,
        seed=seed,
        generator=generator,
        device="cpu",
        dtype=dtype,
        max_attempts_per_obstacle=max_attempts,
    )

    state_after = generator.get_state()

    assert not torch.equal(state_before, state_after)
    assert obstacles.centers.device.type == "cpu"
    assert obstacles.radii.device.type == "cpu"
    assert obstacles.centers.dtype == dtype
    assert obstacles.radii.dtype == dtype
    assert obstacles.centers.shape == (count, 2)
    assert obstacles.radii.shape == (count,)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_public_obstacle_generator_matches_private_helper(dtype):
    config = WorldConfig()
    seed = 7319
    max_attempts = 10_000

    width, height, count, radius_min, radius_max = _validate_inputs(
        config,
        seed,
        max_attempts,
        dtype,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    helper_obstacles = _generate_obstacles_with_rng(
        width=width,
        height=height,
        count=count,
        radius_min=radius_min,
        radius_max=radius_max,
        seed=seed,
        generator=generator,
        device="cpu",
        dtype=dtype,
        max_attempts_per_obstacle=max_attempts,
    )

    public_world = generate_obstacle_world(
        config,
        seed=seed,
        device="cpu",
        dtype=dtype,
        max_attempts_per_obstacle=max_attempts,
    )

    assert torch.equal(
        helper_obstacles.centers,
        public_world.obstacles.centers,
    )
    assert torch.equal(
        helper_obstacles.radii,
        public_world.obstacles.radii,
    )


def test_obstacle_rng_helper_rejects_non_generator():
    with pytest.raises(TypeError, match="torch.Generator"):
        _generate_obstacles_with_rng(
            width=100.0,
            height=100.0,
            count=0,
            radius_min=1.0,
            radius_max=1.0,
            seed=1,
            generator=None,
            device="cpu",
            dtype=torch.float32,
            max_attempts_per_obstacle=10,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_obstacle_rng_helper_cuda_transfer_preserves_cpu_sampled_values(dtype):
    config = WorldConfig()
    seed = 7319
    max_attempts = 10_000

    width, height, count, radius_min, radius_max = _validate_inputs(
        config,
        seed,
        max_attempts,
        dtype,
    )

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed)

    cuda_generator = torch.Generator(device="cpu")
    cuda_generator.manual_seed(seed)

    cpu_obstacles = _generate_obstacles_with_rng(
        width=width,
        height=height,
        count=count,
        radius_min=radius_min,
        radius_max=radius_max,
        seed=seed,
        generator=cpu_generator,
        device="cpu",
        dtype=dtype,
        max_attempts_per_obstacle=max_attempts,
    )

    cuda_obstacles = _generate_obstacles_with_rng(
        width=width,
        height=height,
        count=count,
        radius_min=radius_min,
        radius_max=radius_max,
        seed=seed,
        generator=cuda_generator,
        device="cuda",
        dtype=dtype,
        max_attempts_per_obstacle=max_attempts,
    )

    assert cuda_obstacles.centers.device.type == "cuda"
    assert cuda_obstacles.radii.device.type == "cuda"

    assert torch.equal(
        cpu_obstacles.centers,
        cuda_obstacles.centers.cpu(),
    )
    assert torch.equal(
        cpu_obstacles.radii,
        cuda_obstacles.radii.cpu(),
    )

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_obstacle_rng_helper_respects_caller_advanced_generator_state(dtype):
    # Use one obstacle so the helper accepts its first candidate and consumes
    # exactly three scalar draws: radius, x, then y.
    config = WorldConfig(
        obstacle_count=1,
        obstacle_radius_min_m=1.0,
        obstacle_radius_max_m=1.0,
    )
    validation_seed = 7319
    diagnostic_seed = 999_001
    max_attempts = 10_000

    width, height, count, radius_min, radius_max = _validate_inputs(
        config,
        validation_seed,
        max_attempts,
        dtype,
    )
    assert count == 1

    actual_generator = torch.Generator(device="cpu")
    actual_generator.manual_seed(validation_seed)

    reference_generator = torch.Generator(device="cpu")
    reference_generator.manual_seed(validation_seed)

    # Advance both generators identically before the helper is called.
    torch.rand(
        (37,),
        generator=actual_generator,
        dtype=dtype,
        device="cpu",
    )
    torch.rand(
        (37,),
        generator=reference_generator,
        dtype=dtype,
        device="cpu",
    )

    # Independent oracle: calculate the expected candidate directly from
    # the reference generator, without calling the helper.
    u_radius = torch.rand(
        (),
        generator=reference_generator,
        dtype=dtype,
        device="cpu",
    )
    expected_radius = radius_min + (radius_max - radius_min) * u_radius

    u_x = torch.rand(
        (),
        generator=reference_generator,
        dtype=dtype,
        device="cpu",
    )
    expected_x = expected_radius + u_x * (width - 2.0 * expected_radius)

    u_y = torch.rand(
        (),
        generator=reference_generator,
        dtype=dtype,
        device="cpu",
    )
    expected_y = expected_radius + u_y * (height - 2.0 * expected_radius)

    expected_centers = torch.stack((expected_x, expected_y)).reshape(1, 2)
    expected_radii = expected_radius.reshape(1)

    actual = _generate_obstacles_with_rng(
        width=width,
        height=height,
        count=count,
        radius_min=radius_min,
        radius_max=radius_max,
        seed=diagnostic_seed,
        generator=actual_generator,
        device="cpu",
        dtype=dtype,
        max_attempts_per_obstacle=max_attempts,
    )

    assert torch.equal(actual.centers, expected_centers)
    assert torch.equal(actual.radii, expected_radii)
    assert torch.equal(
        actual_generator.get_state(),
        reference_generator.get_state(),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_obstacle_rng_helper_diagnostic_seed_does_not_change_output(dtype):
    config = WorldConfig()
    generator_seed = 7319
    max_attempts = 10_000

    width, height, count, radius_min, radius_max = _validate_inputs(
        config,
        generator_seed,
        max_attempts,
        dtype,
    )

    generator_a = torch.Generator(device="cpu")
    generator_a.manual_seed(generator_seed)

    generator_b = torch.Generator(device="cpu")
    generator_b.manual_seed(generator_seed)

    # The supplied generator states are identical. Only the diagnostic
    # seed argument differs; it must not seed or otherwise affect sampling.
    actual = _generate_obstacles_with_rng(
        width=width,
        height=height,
        count=count,
        radius_min=radius_min,
        radius_max=radius_max,
        seed=17,
        generator=generator_a,
        device="cpu",
        dtype=dtype,
        max_attempts_per_obstacle=max_attempts,
    )

    comparison = _generate_obstacles_with_rng(
        width=width,
        height=height,
        count=count,
        radius_min=radius_min,
        radius_max=radius_max,
        seed=987_654,
        generator=generator_b,
        device="cpu",
        dtype=dtype,
        max_attempts_per_obstacle=max_attempts,
    )

    assert torch.equal(actual.centers, comparison.centers)
    assert torch.equal(actual.radii, comparison.radii)
    assert torch.equal(generator_a.get_state(), generator_b.get_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_obstacle_rng_helper_zero_count_consumes_no_rng_draws(dtype):
    config = WorldConfig()
    seed = 7319
    max_attempts = 10_000

    width, height, _count, radius_min, radius_max = _validate_inputs(
        config,
        seed,
        max_attempts,
        dtype,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    state_before = generator.get_state().clone()

    obstacles = _generate_obstacles_with_rng(
        width=width,
        height=height,
        count=0,
        radius_min=radius_min,
        radius_max=radius_max,
        seed=seed,
        generator=generator,
        device="cpu",
        dtype=dtype,
        max_attempts_per_obstacle=max_attempts,
    )

    assert obstacles.centers.shape == (0, 2)
    assert obstacles.radii.shape == (0,)
    assert obstacles.centers.dtype == dtype
    assert obstacles.radii.dtype == dtype
    assert torch.equal(generator.get_state(), state_before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_obstacle_rng_helper_rejects_cuda_generator():
    cuda_generator = torch.Generator(device="cuda")

    with pytest.raises(ValueError, match="CPU torch.Generator"):
        _generate_obstacles_with_rng(
            width=100.0,
            height=100.0,
            count=0,
            radius_min=1.0,
            radius_max=1.0,
            seed=1,
            generator=cuda_generator,
            device="cpu",
            dtype=torch.float32,
            max_attempts_per_obstacle=10,
        )
