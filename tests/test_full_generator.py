import pytest
import torch

import lawless.world.full_generator as full_generator
from lawless.world.config import (
    EmbodimentConfig,
    EnvironmentConfig,
    ObjectiveConfig,
    WorldConfig,
)
from lawless.world.full_generator import generate_world
from lawless.world.generated_world import GeneratedWorld
from lawless.world.generator import (
    WorldGenerationError,
    generate_obstacle_world,
)
from lawless.world.geometry import circle_circle_signed_clearance


def _small_config(
    *,
    width=100.0,
    height=100.0,
    obstacle_count=0,
    obstacle_radius_min=1.0,
    obstacle_radius_max=1.0,
    agent_radius=0.5,
    core_count=0,
    core_radius=0.5,
    extraction_radius=1.0,
    seed=42,
):
    return EnvironmentConfig(
        seed=seed,
        world=WorldConfig(
            width_m=width,
            height_m=height,
            obstacle_count=obstacle_count,
            obstacle_radius_min_m=obstacle_radius_min,
            obstacle_radius_max_m=obstacle_radius_max,
        ),
        embodiment=EmbodimentConfig(agent_radius_m=agent_radius),
        objectives=ObjectiveConfig(
            core_count=core_count,
            core_radius_m=core_radius,
            extraction_radius_m=extraction_radius,
        ),
    )


def _assert_strict_clearance(
    centers_a,
    radii_a,
    centers_b,
    radii_b,
    *,
    same_set=False,
):
    if centers_a.shape[0] == 0 or centers_b.shape[0] == 0:
        return

    clearance = circle_circle_signed_clearance(
        centers_a, radii_a, centers_b, radii_b
    )

    if same_set:
        relevant = torch.triu(
            torch.ones_like(clearance, dtype=torch.bool),
            diagonal=1,
        )
        assert bool((clearance[relevant] > 0).all())
    else:
        assert bool((clearance > 0).all())


def _assert_layout_valid(layout: GeneratedWorld):
    world = layout.world
    obstacles = world.obstacles
    dtype = obstacles.centers.dtype
    device = obstacles.centers.device

    spawn_radii = torch.full(
        (layout.spawn_positions.shape[0],),
        layout.agent_body_radius_m,
        dtype=dtype,
        device=device,
    )
    extraction_centers = layout.extraction_center.reshape(1, 2)
    extraction_radii = torch.tensor(
        [layout.extraction_radius_m], dtype=dtype, device=device
    )

    # Every circle is fully contained in the closed world rectangle.
    circle_sets = (
        (obstacles.centers, obstacles.radii),
        (layout.spawn_positions, spawn_radii),
        (extraction_centers, extraction_radii),
        (layout.core_positions, layout.core_radii),
    )
    for centers, radii in circle_sets:
        if centers.shape[0] == 0:
            continue
        x, y = centers.unbind(dim=-1)
        assert bool((x - radii >= 0).all())
        assert bool((x + radii <= world.width_m).all())
        assert bool((y - radii >= 0).all())
        assert bool((y + radii <= world.height_m).all())

    # Strict positive clearance for every required cross-category pair.
    _assert_strict_clearance(
        obstacles.centers, obstacles.radii,
        obstacles.centers, obstacles.radii,
        same_set=True,
    )
    _assert_strict_clearance(
        layout.spawn_positions, spawn_radii,
        obstacles.centers, obstacles.radii,
    )
    _assert_strict_clearance(
        layout.spawn_positions, spawn_radii,
        layout.spawn_positions, spawn_radii,
        same_set=True,
    )
    _assert_strict_clearance(
        extraction_centers, extraction_radii,
        obstacles.centers, obstacles.radii,
    )
    _assert_strict_clearance(
        layout.spawn_positions, spawn_radii,
        extraction_centers, extraction_radii,
    )
    _assert_strict_clearance(
        layout.core_positions, layout.core_radii,
        obstacles.centers, obstacles.radii,
    )
    _assert_strict_clearance(
        layout.core_positions, layout.core_radii,
        layout.spawn_positions, spawn_radii,
    )
    _assert_strict_clearance(
        layout.core_positions, layout.core_radii,
        extraction_centers, extraction_radii,
    )
    _assert_strict_clearance(
        layout.core_positions, layout.core_radii,
        layout.core_positions, layout.core_radii,
        same_set=True,
    )


def _cpu_generator(seed=123):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _patch_candidate_sequence(monkeypatch, centers):
    """Replace candidate draws with a finite, deterministic sequence."""
    sequence = iter(
        torch.tensor(center, dtype=torch.float32)
        for center in centers
    )
    calls = {"count": 0}

    def fake_candidate_center(**kwargs):
        del kwargs
        calls["count"] += 1
        return next(sequence).clone()

    monkeypatch.setattr(
        full_generator,
        "_candidate_center",
        fake_candidate_center,
    )
    return calls


def _patch_repeated_candidate(monkeypatch, center):
    """Return the same candidate indefinitely and count every attempt."""
    candidate = torch.tensor(center, dtype=torch.float32)
    calls = {"count": 0}

    def fake_candidate_center(**kwargs):
        del kwargs
        calls["count"] += 1
        return candidate.clone()

    monkeypatch.setattr(
        full_generator,
        "_candidate_center",
        fake_candidate_center,
    )
    return calls


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_layout_is_reproducible(dtype):
    config = _small_config(
        obstacle_count=8,
        core_count=5,
        seed=1234,
    )

    first = generate_world(config, spawn_count=3, dtype=dtype)
    second = generate_world(config, spawn_count=3, dtype=dtype)

    assert first.seed == second.seed == config.seed
    assert torch.equal(
        first.world.obstacles.centers,
        second.world.obstacles.centers,
    )
    assert torch.equal(
        first.world.obstacles.radii,
        second.world.obstacles.radii,
    )
    assert torch.equal(first.spawn_positions, second.spawn_positions)
    assert torch.equal(first.extraction_center, second.extraction_center)
    assert torch.equal(first.core_positions, second.core_positions)
    assert torch.equal(first.core_radii, second.core_radii)
    _assert_layout_valid(first)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_layout_does_not_change_global_cpu_rng(dtype):
    config = _small_config(obstacle_count=5, core_count=3)

    torch.manual_seed(987654)
    state_before = torch.random.get_rng_state().clone()

    generate_world(config, spawn_count=2, dtype=dtype)

    state_after = torch.random.get_rng_state()
    assert torch.equal(state_before, state_after)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_obstacle_prefix_matches_existing_obstacle_only_generator(dtype):
    config = _small_config(
        obstacle_count=7,
        obstacle_radius_min=0.5,
        obstacle_radius_max=1.5,
        core_count=2,
        seed=2026,
    )

    obstacle_only = generate_obstacle_world(
        config.world,
        seed=config.seed,
        dtype=dtype,
    )
    full = generate_world(config, spawn_count=2, dtype=dtype)

    # Full-layout generation must preserve the historical obstacle stream
    # prefix: no spawn/extraction/core draw occurs before obstacle placement.
    assert torch.equal(
        full.world.obstacles.centers,
        obstacle_only.obstacles.centers,
    )
    assert torch.equal(
        full.world.obstacles.radii,
        obstacle_only.obstacles.radii,
    )


def test_explicit_seed_overrides_config_seed():
    config = _small_config(
        obstacle_count=4,
        core_count=2,
        seed=42,
    )

    default_layout = generate_world(config, spawn_count=2)
    explicit_layout = generate_world(config, seed=991, spawn_count=2)
    same_explicit_layout = generate_world(config, seed=991, spawn_count=2)

    assert default_layout.seed == 42
    assert explicit_layout.seed == 991
    assert torch.equal(
        explicit_layout.world.obstacles.centers,
        same_explicit_layout.world.obstacles.centers,
    )
    assert torch.equal(
        explicit_layout.spawn_positions,
        same_explicit_layout.spawn_positions,
    )


def test_none_seed_uses_environment_config_seed():
    config = _small_config(seed=314159)
    layout = generate_world(config, seed=None)
    assert layout.seed == 314159


def test_spawn_count_is_number_of_distinct_slots():
    config = _small_config()
    layout = generate_world(config, spawn_count=4)

    assert layout.spawn_positions.shape == (4, 2)
    assert layout.seed == config.seed
    _assert_layout_valid(layout)


def test_zero_obstacles_spawns_and_cores_are_supported():
    config = _small_config(
        obstacle_count=0,
        core_count=0,
    )
    layout = generate_world(config, spawn_count=0)

    assert layout.world.obstacles.centers.shape == (0, 2)
    assert layout.world.obstacles.radii.shape == (0,)
    assert layout.spawn_positions.shape == (0, 2)
    assert layout.core_positions.shape == (0, 2)
    assert layout.core_radii.shape == (0,)
    assert layout.extraction_center.shape == (2,)
    _assert_layout_valid(layout)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_configured_radii_are_rounded_to_output_dtype(dtype):
    config = _small_config(
        obstacle_count=1,
        obstacle_radius_min=0.1,
        obstacle_radius_max=0.1,
        agent_radius=0.1,
        core_count=1,
        core_radius=0.1,
        extraction_radius=0.1,
    )
    layout = generate_world(config, dtype=dtype)

    expected = torch.tensor(0.1, dtype=dtype).item()
    assert layout.world.obstacles.radii[0].item() == expected
    assert layout.agent_body_radius_m == expected
    assert layout.extraction_radius_m == expected
    assert layout.core_radii[0].item() == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_layout_tensors_use_requested_dtype_and_device(dtype):
    layout = generate_world(
        _small_config(obstacle_count=3, core_count=2),
        spawn_count=2,
        device="cpu",
        dtype=dtype,
    )

    tensors = (
        layout.world.obstacles.centers,
        layout.world.obstacles.radii,
        layout.spawn_positions,
        layout.extraction_center,
        layout.core_positions,
        layout.core_radii,
    )
    for tensor in tensors:
        assert tensor.dtype == dtype
        assert tensor.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_cuda_full_layout_equality(dtype):
    config = _small_config(
        obstacle_count=8,
        core_count=4,
        seed=8675309,
    )

    cpu = generate_world(
        config, spawn_count=3, device="cpu", dtype=dtype
    )
    cuda = generate_world(
        config, spawn_count=3, device="cuda", dtype=dtype
    )

    assert cuda.world.obstacles.centers.device.type == "cuda"
    assert cuda.spawn_positions.device.type == "cuda"
    assert cuda.extraction_center.device.type == "cuda"
    assert cuda.core_positions.device.type == "cuda"
    assert cuda.core_radii.device.type == "cuda"

    assert torch.equal(
        cpu.world.obstacles.centers,
        cuda.world.obstacles.centers.cpu(),
    )
    assert torch.equal(
        cpu.world.obstacles.radii,
        cuda.world.obstacles.radii.cpu(),
    )
    assert torch.equal(cpu.spawn_positions, cuda.spawn_positions.cpu())
    assert torch.equal(
        cpu.extraction_center,
        cuda.extraction_center.cpu(),
    )
    assert torch.equal(cpu.core_positions, cuda.core_positions.cpu())
    assert torch.equal(cpu.core_radii, cuda.core_radii.cpu())


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"seed": True}, TypeError, "seed"),
        ({"seed": -1}, ValueError, "seed"),
        ({"seed": 1 << 63}, ValueError, "seed"),
        ({"spawn_count": True}, TypeError, "spawn_count"),
        ({"spawn_count": -1}, ValueError, "spawn_count"),
        ({"spawn_count": 1.5}, TypeError, "spawn_count"),
        ({"max_attempts_per_obstacle": 0}, ValueError, "positive"),
        ({"max_attempts_per_spawn": 0}, ValueError, "positive"),
        ({"max_attempts_per_extraction": 0}, ValueError, "positive"),
        ({"max_attempts_per_core": 0}, ValueError, "positive"),
        ({"max_attempts_per_core": True}, TypeError, "integer"),
    ],
)
def test_invalid_arguments_are_rejected(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        generate_world(_small_config(), **kwargs)


def test_unsupported_dtype_is_rejected():
    with pytest.raises(
        TypeError,
        match=r"dtype must be torch\.float32 or torch\.float64",
    ):
        generate_world(
            EnvironmentConfig(),
            seed=7,
            dtype=torch.float16,
        )


def test_invalid_config_type_is_rejected():
    with pytest.raises(TypeError, match="EnvironmentConfig"):
        generate_world(object())


@pytest.mark.parametrize(
    ("agent_radius", "message"),
    [
        (1e-50, "layout dtype"),
        (1e39, "layout dtype"),
    ],
)
def test_agent_radius_unrepresentable_in_float32_is_rejected(
    agent_radius, message
):
    config = _small_config(agent_radius=agent_radius)
    with pytest.raises(ValueError, match=message):
        generate_world(config, dtype=torch.float32)


def test_world_dimension_overflow_in_float32_is_rejected():
    config = _small_config(width=1e39)
    with pytest.raises(ValueError, match="world.width_m"):
        generate_world(config, dtype=torch.float32)


def test_invalid_budget_is_rejected_even_for_zero_count_category():
    config = _small_config(obstacle_count=0, core_count=0)

    with pytest.raises(ValueError, match="max_attempts_per_obstacle"):
        generate_world(
            config,
            spawn_count=0,
            max_attempts_per_obstacle=0,
        )

    with pytest.raises(ValueError, match="max_attempts_per_spawn"):
        generate_world(
            config,
            spawn_count=0,
            max_attempts_per_spawn=0,
        )

    with pytest.raises(ValueError, match="max_attempts_per_core"):
        generate_world(
            config,
            spawn_count=0,
            max_attempts_per_core=0,
        )


def test_obstacle_category_exhaustion_is_bounded_and_identified():
    config = _small_config(
        width=2.0,
        height=2.0,
        obstacle_count=2,
        obstacle_radius_min=1.0,
        obstacle_radius_max=1.0,
    )

    with pytest.raises(WorldGenerationError) as exc_info:
        generate_world(
            config,
            spawn_count=0,
            max_attempts_per_obstacle=1,
        )

    message = str(exc_info.value)
    assert "category='obstacle'" in message
    assert "index=1" in message
    assert "max_attempts=1" in message
    assert "seed=42" in message


def test_spawn_category_exhaustion_is_bounded_and_identified():
    config = _small_config(
        width=2.0,
        height=2.0,
        obstacle_count=1,
        obstacle_radius_min=1.0,
        obstacle_radius_max=1.0,
        agent_radius=0.5,
    )

    with pytest.raises(WorldGenerationError) as exc_info:
        generate_world(
            config,
            spawn_count=1,
            max_attempts_per_spawn=3,
        )

    message = str(exc_info.value)
    assert "category='spawn'" in message
    assert "index=0" in message
    assert "max_attempts=3" in message
    assert "seed=42" in message


def test_extraction_category_exhaustion_is_bounded_and_identified():
    config = _small_config(
        width=2.0,
        height=2.0,
        agent_radius=1.0,
        extraction_radius=1.0,
    )

    with pytest.raises(WorldGenerationError) as exc_info:
        generate_world(
            config,
            spawn_count=1,
            max_attempts_per_extraction=2,
        )

    message = str(exc_info.value)
    assert "category='extraction'" in message
    assert "index=0" in message
    assert "max_attempts=2" in message
    assert "seed=42" in message


def test_core_category_exhaustion_is_bounded_and_identified():
    config = _small_config(
        width=2.0,
        height=2.0,
        core_count=1,
        core_radius=1.0,
        extraction_radius=1.0,
    )

    with pytest.raises(WorldGenerationError) as exc_info:
        generate_world(
            config,
            spawn_count=0,
            max_attempts_per_core=4,
        )

    message = str(exc_info.value)
    assert "category='core'" in message
    assert "index=0" in message
    assert "max_attempts=4" in message
    assert "seed=42" in message


def test_invalid_geometry_is_distinct_from_candidate_exhaustion():
    config = _small_config(
        width=1.0,
        height=1.0,
        agent_radius=0.6,
    )

    with pytest.raises(ValueError, match="spawn body diameter"):
        generate_world(config, spawn_count=1)


def test_full_generator_returns_generated_world_composition():
    layout = generate_world(
        _small_config(obstacle_count=2, core_count=2),
        spawn_count=2,
    )

    assert isinstance(layout, GeneratedWorld)
    assert layout.world.obstacle_count == 2
    assert layout.spawn_positions.shape == (2, 2)
    assert layout.core_positions.shape == (2, 2)


# ---------------------------------------------------------------------------
# Forced candidate tests for strict tangency rejection and budget accounting.
# These exercise the actual placement functions without changing product code.
# ---------------------------------------------------------------------------


def test_spawn_rejects_obstacle_tangency_then_accepts_clear_candidate(
    monkeypatch,
):
    # Obstacle at (5, 5), radius 1. Candidate (7, 5) is exactly tangent
    # for a spawn body of radius 1; candidate (8.1, 5) is strictly clear.
    calls = _patch_candidate_sequence(
        monkeypatch,
        [(7.0, 5.0), (8.1, 5.0)],
    )

    centers, radii = full_generator._place_circle_category(
        category="spawn",
        count=1,
        radius=1.0,
        width=20.0,
        height=20.0,
        generator=_cpu_generator(),
        dtype=torch.float32,
        max_attempts=2,
        seed=17,
        fixed_centers=(
            torch.tensor([[5.0, 5.0]], dtype=torch.float32),
        ),
        fixed_radii=(
            torch.tensor([1.0], dtype=torch.float32),
        ),
    )

    assert calls["count"] == 2
    assert torch.equal(
        centers,
        torch.tensor([[8.1, 5.0]], dtype=torch.float32),
    )
    assert torch.equal(radii, torch.tensor([1.0]))


def test_spawn_rejects_tangency_with_prior_spawn_then_accepts_clear_candidate(
    monkeypatch,
):
    # First spawn is at (2, 2). The second spawn's first candidate is tangent
    # at (4, 2); the next candidate at (6.1, 2) is strictly clear.
    calls = _patch_candidate_sequence(
        monkeypatch,
        [(2.0, 2.0), (4.0, 2.0), (6.1, 2.0)],
    )

    centers, radii = full_generator._place_circle_category(
        category="spawn",
        count=2,
        radius=1.0,
        width=20.0,
        height=20.0,
        generator=_cpu_generator(),
        dtype=torch.float32,
        max_attempts=2,
        seed=18,
        fixed_centers=(),
        fixed_radii=(),
    )

    assert calls["count"] == 3
    assert torch.equal(
        centers,
        torch.tensor(
            [[2.0, 2.0], [6.1, 2.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(radii, torch.tensor([1.0, 1.0]))


def test_extraction_rejects_spawn_body_tangency_then_accepts_clear_candidate(
    monkeypatch,
):
    # This directly verifies the required spawn-body/extraction separation:
    # center distance 2 equals the sum of radii 1 + 1, so the first candidate
    # is rejected. The second has positive clearance.
    calls = _patch_candidate_sequence(
        monkeypatch,
        [(7.0, 5.0), (8.1, 5.0)],
    )

    extraction = full_generator._place_single_circle(
        category="extraction",
        radius=1.0,
        width=20.0,
        height=20.0,
        generator=_cpu_generator(),
        dtype=torch.float32,
        max_attempts=2,
        seed=19,
        other_centers=(
            torch.tensor([[5.0, 5.0]], dtype=torch.float32),
        ),
        other_radii=(
            torch.tensor([1.0], dtype=torch.float32),
        ),
    )

    assert calls["count"] == 2
    assert torch.equal(
        extraction,
        torch.tensor([8.1, 5.0], dtype=torch.float32),
    )

    clearance = circle_circle_signed_clearance(
        extraction.reshape(1, 2),
        torch.tensor([1.0]),
        torch.tensor([[5.0, 5.0]]),
        torch.tensor([1.0]),
    )
    assert bool((clearance > 0).all())


def test_core_rejects_extraction_tangency_then_accepts_clear_candidate(
    monkeypatch,
):
    calls = _patch_candidate_sequence(
        monkeypatch,
        [(7.0, 5.0), (8.1, 5.0)],
    )

    centers, radii = full_generator._place_circle_category(
        category="core",
        count=1,
        radius=1.0,
        width=20.0,
        height=20.0,
        generator=_cpu_generator(),
        dtype=torch.float32,
        max_attempts=2,
        seed=20,
        fixed_centers=(
            torch.tensor([[5.0, 5.0]], dtype=torch.float32),
        ),
        fixed_radii=(
            torch.tensor([1.0], dtype=torch.float32),
        ),
    )

    assert calls["count"] == 2
    assert torch.equal(
        centers,
        torch.tensor([[8.1, 5.0]], dtype=torch.float32),
    )
    assert torch.equal(radii, torch.tensor([1.0]))


@pytest.mark.parametrize(
    ("category", "budget", "center", "expected_index"),
    [
        ("spawn", 3, (7.0, 5.0), 0),
        ("core", 4, (7.0, 5.0), 0),
    ],
)
def test_circle_category_consumes_exact_candidate_budget_on_failure(
    monkeypatch,
    category,
    budget,
    center,
    expected_index,
):
    calls = _patch_repeated_candidate(monkeypatch, center)

    with pytest.raises(WorldGenerationError) as exc_info:
        full_generator._place_circle_category(
            category=category,
            count=1,
            radius=1.0,
            width=20.0,
            height=20.0,
            generator=_cpu_generator(),
            dtype=torch.float32,
            max_attempts=budget,
            seed=21,
            fixed_centers=(
                torch.tensor([[5.0, 5.0]], dtype=torch.float32),
            ),
            fixed_radii=(
                torch.tensor([1.0], dtype=torch.float32),
            ),
        )

    assert calls["count"] == budget
    message = str(exc_info.value)
    assert f"category='{category}'" in message
    assert f"index={expected_index}" in message
    assert f"max_attempts={budget}" in message
    assert "seed=21" in message


def test_extraction_consumes_exact_candidate_budget_on_spawn_tangency_failure(
    monkeypatch,
):
    budget = 5
    calls = _patch_repeated_candidate(monkeypatch, (7.0, 5.0))

    with pytest.raises(WorldGenerationError) as exc_info:
        full_generator._place_single_circle(
            category="extraction",
            radius=1.0,
            width=20.0,
            height=20.0,
            generator=_cpu_generator(),
            dtype=torch.float32,
            max_attempts=budget,
            seed=22,
            other_centers=(
                torch.tensor([[5.0, 5.0]], dtype=torch.float32),
            ),
            other_radii=(
                torch.tensor([1.0], dtype=torch.float32),
            ),
        )

    assert calls["count"] == budget
    message = str(exc_info.value)
    assert "category='extraction'" in message
    assert "index=0" in message
    assert f"max_attempts={budget}" in message
    assert "seed=22" in message


def test_single_circle_candidate_budget_counts_only_actual_candidates(
    monkeypatch,
):
    # First candidate is outside the world and the next two are tangent.
    # All three consume one candidate attempt apiece; no extra hidden retry
    # is permitted after the configured budget is reached.
    budget = 3
    calls = _patch_candidate_sequence(
        monkeypatch,
        [
            (-1.0, 5.0),
            (7.0, 5.0),
            (7.0, 5.0),
        ],
    )

    with pytest.raises(WorldGenerationError):
        full_generator._place_single_circle(
            category="extraction",
            radius=1.0,
            width=20.0,
            height=20.0,
            generator=_cpu_generator(),
            dtype=torch.float32,
            max_attempts=budget,
            seed=23,
            other_centers=(
                torch.tensor([[5.0, 5.0]], dtype=torch.float32),
            ),
            other_radii=(
                torch.tensor([1.0], dtype=torch.float32),
            ),
        )

    assert calls["count"] == budget


def test_circle_category_candidate_budget_counts_each_rejected_candidate(
    monkeypatch,
):
    budget = 4
    calls = _patch_candidate_sequence(
        monkeypatch,
        [
            (-1.0, 5.0),  # Outside the world.
            (7.0, 5.0),   # Tangent to fixed circle.
            (7.0, 5.0),   # Tangent again.
            (7.0, 5.0),   # Final budgeted attempt.
        ],
    )

    with pytest.raises(WorldGenerationError):
        full_generator._place_circle_category(
            category="spawn",
            count=1,
            radius=1.0,
            width=20.0,
            height=20.0,
            generator=_cpu_generator(),
            dtype=torch.float32,
            max_attempts=budget,
            seed=24,
            fixed_centers=(
                torch.tensor([[5.0, 5.0]], dtype=torch.float32),
            ),
            fixed_radii=(
                torch.tensor([1.0], dtype=torch.float32),
            ),
        )

    assert calls["count"] == budget