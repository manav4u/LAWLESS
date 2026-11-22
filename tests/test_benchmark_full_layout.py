import torch

from lawless.world.config import (
    EmbodimentConfig,
    EnvironmentConfig,
    ObjectiveConfig,
    WorldConfig,
)
from scripts.benchmark_full_layout import (
    _run_with_state_capture,
    _same_outcome,
)


def _config(
    *,
    width=100.0,
    height=100.0,
    obstacle_count=0,
    obstacle_radius=1.0,
    extraction_radius=3.0,
    seed=7,
):
    return EnvironmentConfig(
        seed=seed,
        world=WorldConfig(
            width_m=width,
            height_m=height,
            obstacle_count=obstacle_count,
            obstacle_radius_min_m=obstacle_radius,
            obstacle_radius_max_m=obstacle_radius,
        ),
        embodiment=EmbodimentConfig(agent_radius_m=0.5),
        objectives=ObjectiveConfig(
            core_count=0,
            core_radius_m=0.75,
            extraction_radius_m=extraction_radius,
        ),
    )


def _run_instrumented(config, *, obstacle_budget=10):
    return _run_with_state_capture(
        config=config,
        seed=config.seed,
        spawn_count=0,
        dtype=torch.float32,
        device="cpu",
        budgets={
            "obstacle": obstacle_budget,
            "spawn": 10,
            "extraction": 10,
            "core": 10,
        },
        instrument=True,
    )


def test_one_obstacle_without_rejection_reports_one_attempt():
    result = _run_instrumented(
        _config(obstacle_count=1),
        obstacle_budget=10,
    )

    assert result["error_type"] is None

    counts = result["attempt_counts"]["obstacle"]
    assert counts["total"] == 1
    assert counts["by_index"] == {"0": 1}


def test_exhausted_obstacle_budget_reports_exact_attempt_count():
    # The first obstacle is accepted at the only possible center, (1, 1).
    # The second cannot obtain positive clearance and consumes all 3 attempts.
    # Extraction is never reached, but its radius must still fit the world.
    result = _run_instrumented(
        _config(
            width=2.0,
            height=2.0,
            obstacle_count=2,
            obstacle_radius=1.0,
            extraction_radius=0.5,
        ),
        obstacle_budget=3,
    )

    assert result["error_type"] == "WorldGenerationError"
    assert "category='obstacle'" in result["error_message"]
    assert "index=1" in result["error_message"]
    assert "max_attempts=3" in result["error_message"]

    counts = result["attempt_counts"]["obstacle"]
    assert counts["by_index"] == {"0": 1, "1": 3}
    assert counts["total"] == 4


def test_zero_obstacles_reports_zero_attempts():
    result = _run_instrumented(
        _config(obstacle_count=0),
        obstacle_budget=10,
    )

    assert result["error_type"] is None

    counts = result["attempt_counts"]["obstacle"]
    assert counts["total"] == 0
    assert counts["by_index"] == {}


def test_instrumentation_preserves_layout_and_private_rng_state():
    config = _config(
        obstacle_count=1,
        seed=123456,
    )
    budgets = {
        "obstacle": 10,
        "spawn": 10,
        "extraction": 10,
        "core": 10,
    }

    ordinary = _run_with_state_capture(
        config=config,
        seed=config.seed,
        spawn_count=0,
        dtype=torch.float32,
        device="cpu",
        budgets=budgets,
        instrument=False,
    )
    instrumented = _run_with_state_capture(
        config=config,
        seed=config.seed,
        spawn_count=0,
        dtype=torch.float32,
        device="cpu",
        budgets=budgets,
        instrument=True,
    )

    assert ordinary["error_type"] is None
    assert instrumented["error_type"] is None

    assert torch.equal(
        ordinary["result"].world.obstacles.centers,
        instrumented["result"].world.obstacles.centers,
    )
    assert torch.equal(
        ordinary["result"].world.obstacles.radii,
        instrumented["result"].world.obstacles.radii,
    )
    assert torch.equal(
        ordinary["result"].spawn_positions,
        instrumented["result"].spawn_positions,
    )
    assert torch.equal(
        ordinary["result"].extraction_center,
        instrumented["result"].extraction_center,
    )
    assert torch.equal(
        ordinary["result"].core_positions,
        instrumented["result"].core_positions,
    )
    assert torch.equal(
        ordinary["result"].core_radii,
        instrumented["result"].core_radii,
    )

    assert torch.equal(
        ordinary["generator_state"],
        instrumented["generator_state"],
    )

    assert (
        instrumented["attempt_counts"]["obstacle"]["by_index"]
        == {"0": 1}
    )


def test_pre_rng_invalid_geometry_reports_zero_attempts():
    # Extraction cannot fit in this world. Configuration validation rejects
    # it before generate_world creates its private CPU generator.
    config = _config(
        width=2.0,
        height=2.0,
        obstacle_count=2,
        obstacle_radius=1.0,
        extraction_radius=3.0,
    )

    result = _run_instrumented(config, obstacle_budget=3)

    assert result["result"] is None
    assert result["error_type"] == "ValueError"
    assert result["generator_state"] is None
    assert result["attempt_counts"]["obstacle"] == {
        "total": 0,
        "by_index": {},
    }


def test_pre_rng_invalid_geometry_preserves_error_and_rng_equivalence():
    config = _config(
        width=2.0,
        height=2.0,
        obstacle_count=2,
        obstacle_radius=1.0,
        extraction_radius=3.0,
    )
    budgets = {
        "obstacle": 3,
        "spawn": 10,
        "extraction": 10,
        "core": 10,
    }

    ordinary = _run_with_state_capture(
        config=config,
        seed=config.seed,
        spawn_count=0,
        dtype=torch.float32,
        device="cpu",
        budgets=budgets,
        instrument=False,
    )
    instrumented = _run_with_state_capture(
        config=config,
        seed=config.seed,
        spawn_count=0,
        dtype=torch.float32,
        device="cpu",
        budgets=budgets,
        instrument=True,
    )

    assert ordinary["result"] is None
    assert instrumented["result"] is None
    assert ordinary["error_type"] == "ValueError"
    assert instrumented["error_type"] == ordinary["error_type"]
    assert instrumented["error_message"] == ordinary["error_message"]

    assert ordinary["generator_state"] is None
    assert instrumented["generator_state"] is None
    assert _same_outcome(ordinary, instrumented)

    assert instrumented["attempt_counts"]["obstacle"] == {
        "total": 0,
        "by_index": {},
    }


def test_large_obstacle_count_reports_only_attempted_prefix_on_exhaustion():
    config = _config(
        width=2.0,
        height=2.0,
        obstacle_count=20,
        obstacle_radius=1.0,
        extraction_radius=0.5,
    )

    result = _run_instrumented(
        config,
        obstacle_budget=3,
    )

    assert result["result"] is None
    assert result["error_type"] == "WorldGenerationError"
    assert "category='obstacle'" in result["error_message"]
    assert "index=1" in result["error_message"]
    assert "max_attempts=3" in result["error_message"]

    counts = result["attempt_counts"]["obstacle"]
    assert counts["total"] == 4
    assert counts["by_index"] == {
        "0": 1,
        "1": 3,
    }
