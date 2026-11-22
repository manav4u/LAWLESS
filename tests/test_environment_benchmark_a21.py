
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from benchmarks import environment_benchmark as bench


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_a21_config_uses_locked_full_episode_and_shift():
    config = bench.create_a21_config(7)

    assert config.episode.episode_duration_s == 40.0
    assert config.episode.shift_time_s == 20.0
    assert config.episode.dt_s == pytest.approx(1.0 / 60.0)
    assert config.episode.gravity_m_s2 == 4.0

    assert round(
        config.episode.episode_duration_s / config.episode.dt_s
    ) == 2400
    assert round(
        config.episode.shift_time_s / config.episode.dt_s
    ) == 1200


def test_a21_config_rejects_changed_episode_timing(monkeypatch):
    config = bench.EnvironmentConfig()
    config.seed = 7
    config.episode.shift_time_s = 10.0
    config.validate()

    monkeypatch.setattr(
        bench,
        "create_config",
        lambda seed: config,
    )

    with pytest.raises(
        ValueError,
        match="A2.1 requires episode.shift_time_s",
    ):
        bench.create_a21_config(7)


def test_a21_config_rejects_zero_gravity(monkeypatch):
    config = bench.EnvironmentConfig()
    config.seed = 7
    config.episode.gravity_m_s2 = 0.0

    monkeypatch.setattr(
        bench,
        "create_config",
        lambda seed: config,
    )

    with pytest.raises(
        ValueError,
        match="gravity_m_s2 must be positive",
    ):
        bench.create_a21_config(7)


@pytest.mark.parametrize(
    "field,value",
    [
        ("obstacle_count", 0),
        ("core_count", 0),
        ("spawn_count", 0),
    ],
)
def test_a21_geometry_rejects_empty_required_geometry(field, value):
    inventory = [
        {
            "obstacle_count": 20,
            "core_count": 8,
            "spawn_count": 1,
            "extraction_radius_m": 3.0,
            "agent_body_radius_m": 0.5,
        }
    ]
    inventory[0][field] = value

    with pytest.raises(ValueError, match="A2.1 world 0"):
        bench.validate_a21_geometry(inventory, batch=1)


def test_a21_geometry_requires_one_inventory_per_world():
    inventory = [
        {
            "obstacle_count": 20,
            "core_count": 8,
            "spawn_count": 1,
            "extraction_radius_m": 3.0,
            "agent_body_radius_m": 0.5,
        }
    ]

    with pytest.raises(
        ValueError,
        match="expected geometry for 2 worlds",
    ):
        bench.validate_a21_geometry(inventory, batch=2)


def test_a21_fixture_rejects_non_cpu_device_before_execution(tmp_path):
    output = tmp_path / "a21.json"

    with pytest.raises(
        ValueError,
        match="unsupported benchmark device type",
    ):
        bench.execute_a21_stress_fixture(
            root=REPO_ROOT,
            batch=1,
            candidates=1,
            seed=7,
            device=torch.device("meta"),
            dtype=torch.float32,
            output_path=output,
        )

    assert not output.exists()


def test_a21_real_cpu_full_episode_stress_fixture(tmp_path):
    output = tmp_path / "a21_full_episode.json"

    result = bench.execute_a21_stress_fixture(
        root=REPO_ROOT,
        batch=1,
        candidates=1,
        seed=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
        output_path=output,
    )

    assert result["status"] == "completed"
    assert result["termination_reason"] == "episode_deadline"
    assert result["protocol"] == "P1.8-A2.1"

    assert result["expected_world_transitions_per_world"] == 2400
    assert result["expected_gravity_shift_transition"] == 1200

    assert result["accounting"]["step_calls"] == 2400
    assert result["accounting"]["world_step_transitions"] == 2400
    assert result["accounting"]["candidate_slot_transitions"] == 2400
    assert result["accounting"]["per_world_step_transitions"] == [2400]
    assert result["accounting"]["newly_completed_worlds"] == 1

    assert result["final_inventory"]["worlds_completed"] == 1
    assert result["final_inventory"]["world_step_counts"] == [2400]

    assert result["geometry_validation_scope"] == (
        "every successfully returned transition"
    )
    assert result["geometry_validation_checks"] == 2400
    assert result["geometry_world_validation_checks"] == 2400
    assert result["geometry_last_checked_transition"] == 2400
    assert result["geometry_validation_complete"] is True

    initial_geometry = result["initial_geometry_inventory"]
    final_geometry = result["final_geometry_inventory"]

    assert len(initial_geometry) == 1
    assert initial_geometry[0]["obstacle_count"] > 0
    assert initial_geometry[0]["core_count"] > 0
    assert initial_geometry[0]["spawn_count"] > 0
    assert final_geometry == initial_geometry

    shift = result["gravity_shift_observation"]
    assert shift is not None
    assert shift["transition"] == 1200
    assert shift["elapsed_time_s"] == pytest.approx(20.0)
    assert shift["shift_event_per_world"] == [1]
    assert shift["persistent_shift_applied_per_world"] == [True]

    gravity_before = torch.tensor(
        shift["gravity_used_first_world"],
        dtype=torch.float64,
    )
    gravity_after = torch.tensor(
        shift["gravity_next_first_world"],
        dtype=torch.float64,
    )

    assert torch.linalg.vector_norm(gravity_before).item() == pytest.approx(
        4.0, rel=1e-5, abs=1e-5
    )
    assert torch.linalg.vector_norm(gravity_after).item() == pytest.approx(
        4.0, rel=1e-5, abs=1e-5
    )
    assert not torch.allclose(gravity_before, gravity_after)

    assert output.is_file()

    import json

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "completed"
    assert saved["accounting"]["world_step_transitions"] == 2400
    assert saved["gravity_shift_observation"]["transition"] == 1200

    assert saved["geometry_validation_checks"] == 2400
    assert saved["geometry_world_validation_checks"] == 2400
    assert saved["geometry_last_checked_transition"] == 2400
    assert saved["geometry_validation_complete"] is True

    saved_shift = saved["gravity_shift_observation"]
    assert saved_shift["shift_event_per_world"] == [1]
    assert saved_shift["persistent_shift_applied_per_world"] == [True]
    assert saved_shift["actual_world_shift_applied_per_world"] == [True]
    assert saved_shift["actual_world_step_counts"] == [1200]
    assert saved_shift["post_shift_world_gravity_vectors"] == (
        result["gravity_shift_observation"]["post_shift_world_gravity_vectors"]
    )
