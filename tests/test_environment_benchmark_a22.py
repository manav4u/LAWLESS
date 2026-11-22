"""Focused CPU tests for the A2.2 environment timing runner.

The synthetic tests isolate runner control flow and validation gates. They do
not constitute production workload evidence. The final smoke test is the one
unmocked seed-7, B=1, N=1, 2400-step production execution.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from benchmarks import environment_timing_a22 as a22
from benchmarks import environment_benchmark as a21


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------

BATCH = 1
CANDIDATES = 1
OBSERVATION_WIDTH = 23


class FakePhysics:
    def __init__(
        self,
        candidates: int = CANDIDATES,
        *,
        alive_shape: tuple[int, ...] | None = None,
        nonfinite_field: str | None = None,
    ) -> None:
        self.position = torch.zeros((candidates, 2), dtype=torch.float32)
        self.velocity = torch.zeros((candidates, 2), dtype=torch.float32)
        self.angle = torch.zeros((candidates, 1), dtype=torch.float32)
        self.angular_velocity = torch.zeros(
            (candidates, 1), dtype=torch.float32
        )
        self.energy = torch.ones((candidates, 1), dtype=torch.float32)

        if alive_shape is None:
            alive_shape = (candidates, 1)
        self.alive = torch.ones(alive_shape, dtype=torch.bool)

        if nonfinite_field is not None:
            value = getattr(self, nonfinite_field)
            if value.is_floating_point():
                value.reshape(-1)[0] = float("nan")

    def validate_shapes(self) -> None:
        expected = {
            "position": (self.position.shape[0], 2),
            "velocity": (self.velocity.shape[0], 2),
            "angle": (self.position.shape[0], 1),
            "angular_velocity": (self.position.shape[0], 1),
            "energy": (self.position.shape[0], 1),
            "alive": (self.position.shape[0], 1),
        }
        for name, shape in expected.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f"{name} has invalid shape")


def make_physical_environment(
    *,
    batch: int = BATCH,
    candidates: int = CANDIDATES,
    alive_shape: tuple[int, ...] | None = None,
    nonfinite_field: str | None = None,
) -> Any:
    worlds = []
    for _ in range(batch):
        physics = FakePhysics(
            candidates,
            alive_shape=alive_shape,
            nonfinite_field=nonfinite_field,
        )
        worlds.append(
            SimpleNamespace(
                episode=SimpleNamespace(physics=physics),
                gravity=torch.tensor([0.0, 4.0], dtype=torch.float32),
            )
        )
    return SimpleNamespace(worlds=tuple(worlds))


def valid_final_inventory(
    *,
    batch: int = BATCH,
    candidates: int = CANDIDATES,
) -> dict[str, Any]:
    return {
        "physical_alive_per_world": [candidates] * batch,
        "extracted_per_world": [0] * batch,
        "failed_per_world": [0] * batch,
        "truncated_per_world": [candidates] * batch,
    }


def fake_source_identity(revision: str = "synthetic-revision") -> dict:
    return {
        "status": "verified",
        "identity_complete": True,
        "revision": revision,
        "branch": "synthetic",
        "benchmark_included": True,
    }


def fake_artifact_hashes() -> dict[str, dict[str, str]]:
    return {
        "runner": {"status": "verified", "sha256": "a" * 64},
        "tests": {"status": "verified", "sha256": "b" * 64},
    }


def install_synthetic_execute_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    loop_steps: int = 2,
    step_failure_at: int | None = None,
    final_validation_error: Exception | None = None,
    event_log: list[str] | None = None,
    timer_values: list[float] | None = None,
) -> dict[str, Any]:
    """Install a wholly synthetic execution environment for runner tests."""
    events = event_log if event_log is not None else []
    environment = SimpleNamespace(
        worlds=(object(),),
        num_agents=CANDIDATES,
        config=object(),
        all_completed=True,
    )
    observations = torch.zeros(
        (BATCH, CANDIDATES, OBSERVATION_WIDTH),
        dtype=torch.float32,
        device="cpu",
    )
    targets = torch.zeros((BATCH, CANDIDATES, 2), dtype=torch.float32)
    config = object()
    step_calls = {"count": 0}
    identity = fake_source_identity()

    monkeypatch.setattr(a22, "A21_EPISODE_STEPS", loop_steps)
    monkeypatch.setattr(a22, "_get_source_identity", lambda _root: identity)
    monkeypatch.setattr(a22, "_artifact_hashes", lambda _root: fake_artifact_hashes())
    monkeypatch.setattr(a22, "_validate_config", lambda _config: None)
    monkeypatch.setattr(
        a22,
        "_validate_observations",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        a22,
        "_validate_initial_environment",
        lambda *_args, **_kwargs: {
            "initial_gravity": [[0.0, 4.0]],
            "geometry": {"synthetic": True},
        },
    )
    monkeypatch.setattr(
        a22,
        "_capture_target_positions",
        lambda _environment: targets,
    )

    def reset(*_args: Any, **_kwargs: Any) -> tuple[Any, Any, dict]:
        events.append("reset")
        return environment, observations, {"synthetic_reset": True}

    def controller(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        events.append("controller")
        return torch.zeros((BATCH, CANDIDATES, 2), dtype=torch.float32)

    def step(
        current_environment: Any,
        _actions: Any,
    ) -> tuple[Any, Any, Any, Any, Any, dict]:
        step_calls["count"] += 1
        events.append("step")
        if (
            step_failure_at is not None
            and step_calls["count"] == step_failure_at
        ):
            raise RuntimeError("synthetic step interruption")
        return (
            current_environment,
            observations,
            torch.zeros((BATCH, CANDIDATES)),
            torch.zeros((BATCH, CANDIDATES), dtype=torch.bool),
            torch.zeros((BATCH, CANDIDATES), dtype=torch.bool),
            {},
        )

    def validate_final(*_args: Any, **_kwargs: Any) -> dict:
        events.append("final_validation")
        if final_validation_error is not None:
            raise final_validation_error
        return {"synthetic_final_validation": True}

    monkeypatch.setattr(a21, "create_a21_config", lambda _seed: config)
    monkeypatch.setattr(a21, "reset", reset)
    monkeypatch.setattr(a21, "a21_controller_actions", controller)
    monkeypatch.setattr(a21, "step", step)
    monkeypatch.setattr(a22, "_validate_final_environment", validate_final)

    if timer_values is not None:
        values = iter(timer_values)

        def fake_perf_counter() -> float:
            events.append("timer")
            return next(values)

        monkeypatch.setattr(a22.time, "perf_counter", fake_perf_counter)

    return {
        "environment": environment,
        "observations": observations,
        "targets": targets,
        "events": events,
        "step_calls": step_calls,
        "identity": identity,
    }


def run_synthetic_repetition(
    root: Path,
    *,
    seed: int = 7,
    phase: str = "measured",
) -> dict[str, Any]:
    return a22.execute_repetition(
        root=root,
        seed=seed,
        batch=BATCH,
        candidates=CANDIDATES,
        repetition_index=0,
        phase=phase,
    )


# ---------------------------------------------------------------------------
# Observation contract
# ---------------------------------------------------------------------------

def test_observation_contract_accepts_exact_cpu_float32_shape() -> None:
    observations = torch.zeros(
        (2, 3, 23),
        dtype=torch.float32,
        device="cpu",
    )

    a22._validate_observations(
        observations,
        batch=2,
        candidates=3,
        label="test",
    )


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 22),
        (1, 1, 24),
        (1, 23),
        (1, 1, 23, 1),
        (2, 1, 23),
    ],
)
def test_observation_contract_rejects_wrong_shape(shape: tuple[int, ...]) -> None:
    observations = torch.zeros(shape, dtype=torch.float32)

    with pytest.raises(a22.A22InvalidRun):
        a22._validate_observations(
            observations,
            batch=1,
            candidates=1,
            label="test",
        )


def test_observation_contract_rejects_non_float32() -> None:
    observations = torch.zeros((1, 1, 23), dtype=torch.float64)

    with pytest.raises(a22.A22InvalidRun):
        a22._validate_observations(
            observations,
            batch=1,
            candidates=1,
            label="test",
        )


def test_observation_contract_rejects_nonfinite_values() -> None:
    observations = torch.zeros((1, 1, 23), dtype=torch.float32)
    observations[0, 0, 0] = float("nan")

    with pytest.raises(a22.A22InvalidRun):
        a22._validate_observations(
            observations,
            batch=1,
            candidates=1,
            label="test",
        )


def test_observation_contract_rejects_non_cpu_device_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not required for the CPU test suite")

    observations = torch.zeros(
        (1, 1, 23),
        dtype=torch.float32,
        device="cuda",
    )

    with pytest.raises(a22.A22InvalidRun):
        a22._validate_observations(
            observations,
            batch=1,
            candidates=1,
            label="test",
        )


# ---------------------------------------------------------------------------
# Final physical-state and deadline inventory checks
# ---------------------------------------------------------------------------

def test_final_physical_inventory_accepts_production_alive_shape_n_by_one() -> None:
    environment = make_physical_environment()
    inventory = valid_final_inventory()

    a22._validate_final_physical_inventory(
        environment,
        inventory,
        batch=1,
        candidates=1,
    )


def test_final_physical_inventory_rejects_alive_shape_n() -> None:
    environment = make_physical_environment(alive_shape=(1,))
    inventory = valid_final_inventory()

    with pytest.raises(a22.A22InvalidRun):
        a22._validate_final_physical_inventory(
            environment,
            inventory,
            batch=1,
            candidates=1,
        )


@pytest.mark.parametrize(
    ("field", "values"),
    [
        ("physical_alive_per_world", [0]),
        ("physical_alive_per_world", [2]),
        ("extracted_per_world", [1]),
        ("failed_per_world", [1]),
        ("truncated_per_world", [0]),
    ],
)
def test_final_physical_inventory_rejects_incorrect_deadline_counts(
    field: str,
    values: list[int],
) -> None:
    environment = make_physical_environment()
    inventory = valid_final_inventory()
    inventory[field] = values

    with pytest.raises(a22.A22InvalidRun):
        a22._validate_final_physical_inventory(
            environment,
            inventory,
            batch=1,
            candidates=1,
        )


def test_final_physical_inventory_rejects_noninteger_counts() -> None:
    environment = make_physical_environment()
    inventory = valid_final_inventory()
    inventory["truncated_per_world"] = [True]

    with pytest.raises(a22.A22InvalidRun):
        a22._validate_final_physical_inventory(
            environment,
            inventory,
            batch=1,
            candidates=1,
        )


@pytest.mark.parametrize(
    "field",
    [
        "position",
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
    ],
)
def test_final_physical_inventory_rejects_nonfinite_physics(
    field: str,
) -> None:
    environment = make_physical_environment(nonfinite_field=field)

    with pytest.raises(a22.A22InvalidRun):
        a22._validate_final_physical_inventory(
            environment,
            valid_final_inventory(),
            batch=1,
            candidates=1,
        )


def test_final_physical_inventory_rejects_dead_candidate() -> None:
    environment = make_physical_environment()
    environment.worlds[0].episode.physics.alive[0, 0] = False

    with pytest.raises(a22.A22InvalidRun):
        a22._validate_final_physical_inventory(
            environment,
            valid_final_inventory(),
            batch=1,
            candidates=1,
        )


def test_final_physical_inventory_requires_production_shape_validator() -> None:
    environment = make_physical_environment()
    environment.worlds[0].episode.physics.validate_shapes = None

    with pytest.raises(a22.A22InvalidRun):
        a22._validate_final_physical_inventory(
            environment,
            valid_final_inventory(),
            batch=1,
            candidates=1,
        )


# ---------------------------------------------------------------------------
# Partial progress: per-world state is not public-call accounting
# ---------------------------------------------------------------------------

def test_partial_progress_reports_world_counts_but_not_public_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = SimpleNamespace(
        all_completed=False,
        worlds=(object(), object()),
    )

    monkeypatch.setattr(a21, "per_world_steps", lambda _env: [17, 12])
    monkeypatch.setattr(
        a21,
        "per_world_completed",
        lambda _env: [False, False],
    )
    monkeypatch.setattr(
        a21,
        "final_inventory",
        lambda _env: {"partial": True},
    )

    progress = a22._partial_progress(environment)

    assert progress["world_step_counts"] == [17, 12]
    assert progress["minimum_world_step_count"] == 12
    assert progress["world_completed"] == [False, False]
    assert progress["all_completed"] is False
    assert progress["transitions_attempted"] is None
    assert progress["public_step_calls_returned"] is None
    assert progress["partial_final_inventory"] == {"partial": True}


def test_partial_progress_handles_unavailable_state_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = object()

    def fail_snapshot(_environment: Any) -> list[int]:
        raise RuntimeError("synthetic snapshot failure")

    monkeypatch.setattr(a21, "per_world_steps", fail_snapshot)

    progress = a22._partial_progress(environment)

    assert progress["world_step_counts"] is None
    assert progress["minimum_world_step_count"] is None
    assert progress["transitions_attempted"] is None
    assert progress["public_step_calls_returned"] is None
    assert "progress_snapshot_error" in progress


def test_partial_progress_of_missing_environment_is_unknown() -> None:
    progress = a22._partial_progress(None)

    assert progress["transitions_attempted"] is None
    assert progress["public_step_calls_returned"] is None
    assert progress["world_step_counts"] is None


# ---------------------------------------------------------------------------
# Timer boundaries, setup/reset isolation, and receipt counts
# ---------------------------------------------------------------------------

def test_synthetic_timer_boundaries_exclude_setup_reset_and_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    # setup: 1 -> 1 (zero); reset: 2 -> 2 (zero);
    # episode: 3 -> 5 (positive).
    fixture = install_synthetic_execute_dependencies(
        monkeypatch,
        loop_steps=2,
        event_log=events,
        timer_values=[1.0, 1.0, 2.0, 2.0, 3.0, 5.0],
    )

    receipt = run_synthetic_repetition(tmp_path)

    assert receipt["valid"] is True
    assert receipt["setup_seconds"] == 0.0
    assert receipt["reset_seconds"] == 0.0
    assert receipt["episode_seconds"] == 2.0
    assert receipt["transitions_attempted"] == 2
    assert receipt["public_step_calls_returned"] == 2
    assert fixture["step_calls"]["count"] == 2

    # The first episode timer edge is immediately before controller work.
    # The second is immediately after the last public step returns.
    episode_timer_positions = [
        index
        for index, item in enumerate(events)
        if item == "timer"
    ]
    assert len(episode_timer_positions) == 6
    assert events[episode_timer_positions[4] + 1] == "controller"
    assert events[episode_timer_positions[5] - 1] == "step"

    # Setup and reset have completed before the episode timer starts.
    assert events.index("reset") < episode_timer_positions[4]

    # No per-transition diagnostic callback is inserted in the timed loop.
    timed_events = events[
        episode_timer_positions[4] : episode_timer_positions[5] + 1
    ]
    assert timed_events == [
        "timer",
        "controller",
        "step",
        "controller",
        "step",
        "timer",
    ]


def test_synthetic_interrupted_loop_keeps_public_call_count_unknown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture = install_synthetic_execute_dependencies(
        monkeypatch,
        loop_steps=4,
        step_failure_at=2,
    )

    monkeypatch.setattr(a21, "per_world_steps", lambda _env: [1])
    monkeypatch.setattr(
        a21,
        "per_world_completed",
        lambda _env: [False],
    )
    monkeypatch.setattr(
        a21,
        "final_inventory",
        lambda _env: {"partial": True},
    )

    receipt = run_synthetic_repetition(tmp_path)

    assert receipt["valid"] is False
    assert receipt["transitions_attempted"] is None
    assert receipt["public_step_calls_returned"] is None
    assert receipt["partial_progress"]["world_step_counts"] == [1]
    assert receipt["partial_progress"]["minimum_world_step_count"] == 1
    assert receipt["partial_progress"]["public_step_calls_returned"] is None
    assert fixture["step_calls"]["count"] == 2


def test_synthetic_normal_loop_preserves_exact_counts_after_later_rejection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    loop_steps = 3
    fixture = install_synthetic_execute_dependencies(
        monkeypatch,
        loop_steps=loop_steps,
        final_validation_error=RuntimeError("synthetic final-state rejection"),
    )

    receipt = run_synthetic_repetition(tmp_path)

    assert receipt["valid"] is False
    assert receipt["transitions_attempted"] == loop_steps
    assert receipt["public_step_calls_returned"] == loop_steps
    assert fixture["step_calls"]["count"] == loop_steps
    assert receipt["episode_seconds"] is None
    assert receipt["episode_steps_per_second"] is None
    assert receipt["world_steps_per_second"] is None


@pytest.mark.parametrize(
    ("timer_values", "expected_valid"),
    [
        ([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], False),
        ([1.0, 1.0, 2.0, 2.0, 3.0, 2.0], False),
        ([1.0, 1.0, 2.0, 2.0, 3.0, float("inf")], False),
        ([1.0, 1.0, 2.0, 2.0, 3.0, float("nan")], False),
    ],
)
def test_synthetic_nonpositive_or_nonfinite_episode_interval_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    timer_values: list[float],
    expected_valid: bool,
) -> None:
    install_synthetic_execute_dependencies(
        monkeypatch,
        loop_steps=2,
        timer_values=timer_values,
    )

    receipt = run_synthetic_repetition(tmp_path)

    assert receipt["valid"] is expected_valid
    assert receipt["episode_seconds"] is None
    assert receipt["episode_steps_per_second"] is None
    assert receipt["world_steps_per_second"] is None
    assert receipt["transitions_attempted"] == 2
    assert receipt["public_step_calls_returned"] == 2


def test_synthetic_failed_receipt_is_strict_json_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_synthetic_execute_dependencies(
        monkeypatch,
        loop_steps=2,
        timer_values=[
            1.0,
            1.0,
            2.0,
            2.0,
            3.0,
            float("nan"),
        ],
    )

    receipt = run_synthetic_repetition(tmp_path)

    assert receipt["valid"] is False
    assert receipt["episode_seconds"] is None
    encoded = json.dumps(receipt, allow_nan=False)
    assert isinstance(encoded, str)


def test_synthetic_zero_setup_and_reset_are_not_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_synthetic_execute_dependencies(
        monkeypatch,
        loop_steps=1,
        timer_values=[1.0, 1.0, 2.0, 2.0, 3.0, 4.0],
    )

    receipt = run_synthetic_repetition(tmp_path)

    assert receipt["setup_seconds"] == 0.0
    assert receipt["reset_seconds"] == 0.0
    assert receipt["episode_seconds"] == 1.0
    assert receipt["valid"] is True


# ---------------------------------------------------------------------------
# Frozen P0 preservation and source identity
# ---------------------------------------------------------------------------

def make_source_inventory(
    *,
    frozen_count: int = 20,
    mismatch_index: int | None = None,
    missing_current_index: int | None = None,
) -> dict[str, Any]:
    frozen_entries = {}
    current_entries = {}

    for index in range(frozen_count):
        path = f"lawless/frozen/p0_file_{index:02d}.py"
        digest = f"{index + 1:064x}"
        frozen_entries[path] = digest

        if index != missing_current_index:
            current_entries[path] = digest

    if mismatch_index is not None:
        path = f"lawless/frozen/p0_file_{mismatch_index:02d}.py"
        current_entries[path] = "f" * 64

    current_entries["benchmarks/environment_benchmark.py"] = "a" * 64
    current_entries["benchmarks/environment_timing_a22.py"] = "b" * 64

    return {
        "status": "verified",
        "identity_complete": True,
        "revision": "synthetic-source-revision",
        "branch": "synthetic-branch",
        "benchmark_included": True,
        "current_python_hashes": {
            "status": "verified",
            "entries": current_entries,
        },
        "frozen_p0_python_hashes": {
            "status": "verified",
            "entries": frozen_entries,
        },
    }


def test_source_identity_accepts_exact_twenty_matching_frozen_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = make_source_inventory()
    monkeypatch.setattr(a21, "source_identity", lambda _root: identity)

    result = a22._get_source_identity(tmp_path)

    assert result["frozen_p0_preservation"]["status"] == "verified"
    assert result["frozen_p0_preservation"]["file_count"] == 20
    assert result["frozen_p0_preservation"]["compared_file_count"] == 20
    assert result["frozen_p0_preservation"]["mismatched_files"] == []


@pytest.mark.parametrize(
    "identity",
    [
        make_source_inventory(frozen_count=19),
        make_source_inventory(mismatch_index=4),
        make_source_inventory(missing_current_index=7),
    ],
)
def test_source_identity_rejects_incomplete_or_changed_frozen_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    identity: dict[str, Any],
) -> None:
    monkeypatch.setattr(a21, "source_identity", lambda _root: identity)

    with pytest.raises(a22.A22InvalidRun):
        a22._get_source_identity(tmp_path)


def test_source_identity_rejects_incomplete_checkout_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = make_source_inventory()
    identity["identity_complete"] = False
    monkeypatch.setattr(a21, "source_identity", lambda _root: identity)

    with pytest.raises(a22.A22InvalidRun):
        a22._get_source_identity(tmp_path)


def test_source_identity_rejects_missing_benchmark_from_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = make_source_inventory()
    identity["benchmark_included"] = False
    monkeypatch.setattr(a21, "source_identity", lambda _root: identity)

    with pytest.raises(a22.A22InvalidRun):
        a22._get_source_identity(tmp_path)


def test_source_fingerprint_detects_before_after_change() -> None:
    before = fake_source_identity("revision-before")
    after = fake_source_identity("revision-after")

    assert a22._identity_fingerprint(before) != (
        a22._identity_fingerprint(after)
    )


def test_source_fingerprint_requires_before_after_and_cross_series_match() -> None:
    matching = a22._identity_fingerprint(fake_source_identity())
    repetitions = [
        {
            "valid": True,
            "source_fingerprint_before": matching,
            "source_fingerprint_after": matching,
        },
        {
            "valid": True,
            "source_fingerprint_before": matching,
            "source_fingerprint_after": matching,
        },
    ]

    assert a22._common_source_fingerprint(repetitions) == matching

    changed_after = [dict(item) for item in repetitions]
    changed_after[1]["source_fingerprint_after"] = "changed"
    with pytest.raises(a22.A22InvalidRun):
        a22._common_source_fingerprint(changed_after)

    changed_series = [dict(item) for item in repetitions]
    changed_series[1]["source_fingerprint_before"] = "other-source"
    changed_series[1]["source_fingerprint_after"] = "other-source"
    with pytest.raises(a22.A22InvalidRun):
        a22._common_source_fingerprint(changed_series)


# ---------------------------------------------------------------------------
# Aggregate summary and withheld rates
# ---------------------------------------------------------------------------

def make_valid_repetition(
    *,
    seconds: float = 2.0,
    fingerprint: str = "same-source",
    batch: int = 1,
) -> dict[str, Any]:
    return {
        "valid": True,
        "status": "valid",
        "batch": batch,
        "episode_seconds": seconds,
        "episode_steps_per_second": a22.A21_EPISODE_STEPS / seconds,
        "world_steps_per_second": (
            a22.A21_EPISODE_STEPS * batch / seconds
        ),
        "source_fingerprint_before": fingerprint,
        "source_fingerprint_after": fingerprint,
    }


def test_summary_uses_only_finite_positive_valid_episode_times() -> None:
    repetitions = [
        make_valid_repetition(seconds=2.0),
        make_valid_repetition(seconds=4.0),
        {
            "valid": False,
            "batch": 1,
            "episode_seconds": None,
            "episode_steps_per_second": None,
            "world_steps_per_second": None,
        },
    ]

    summary = a22._summarize_repetitions(repetitions)

    assert summary["all_measured_repetitions_valid"] is False
    assert summary["median_episode_seconds"] is None
    assert summary["median_episode_steps_per_second"] is None
    assert summary["median_world_steps_per_second"] is None


@pytest.mark.parametrize(
    "bad_seconds",
    [0.0, -1.0, float("inf"), float("nan")],
)
def test_summary_withholds_rates_for_invalid_raw_duration(
    bad_seconds: float,
) -> None:
    repetitions = [
        make_valid_repetition(seconds=2.0),
        {
            "valid": False,
            "batch": 1,
            "episode_seconds": bad_seconds,
            "episode_steps_per_second": None,
            "world_steps_per_second": None,
        },
    ]

    summary = a22._summarize_repetitions(repetitions)

    assert summary["all_measured_repetitions_valid"] is False
    assert summary["median_episode_seconds"] is None
    assert summary["median_episode_steps_per_second"] is None
    assert summary["median_world_steps_per_second"] is None


def _invoke_run_benchmark_with_supported_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Any:
    """Call run_benchmark using its declared parameter names and defaults."""
    signature = inspect.signature(a22.run_benchmark)
    supplied: dict[str, Any] = {}

    candidate_values = {
        "root": tmp_path,
        "seed": 7,
        "batch": 1,
        "candidates": 1,
        "warmups": 1,
        "warmup_repetitions": 1,
        "repetitions": 2,
        "measured_repetitions": 2,
        "output": None,
        "output_path": None,
        "device": "cpu",
    }

    for name, parameter in signature.parameters.items():
        if name in candidate_values:
            supplied[name] = candidate_values[name]
        elif parameter.default is inspect.Parameter.empty:
            raise AssertionError(
                "Test needs an explicit value for run_benchmark parameter "
                f"{name!r}; update candidate_values for the runner API."
            )

    # Synthetic repetition receipts carry consistent source fingerprints.
    fingerprint = "synthetic-common-fingerprint"
    synthetic_receipt = make_valid_repetition(fingerprint=fingerprint)

    monkeypatch.setattr(
        a22,
        "execute_repetition",
        lambda **kwargs: dict(
            synthetic_receipt,
            phase=kwargs.get("phase", "measured"),
            repetition_index=kwargs.get("repetition_index", 0),
        ),
    )
    monkeypatch.setattr(
        a22,
        "_runtime_metadata",
        lambda *_args, **_kwargs: {"device": "cpu", "synthetic": True},
    )

    # The first artifact snapshot is verified; the second differs. This
    # simulates a source/artifact change across the benchmark series.
    artifact_calls = {"count": 0}

    def artifact_snapshot(_root: Path) -> dict[str, dict[str, str]]:
        artifact_calls["count"] += 1
        result = fake_artifact_hashes()
        if artifact_calls["count"] >= 2:
            result["runner"]["sha256"] = "c" * 64
        return result

    monkeypatch.setattr(a22, "_artifact_hashes", artifact_snapshot)

    return a22.run_benchmark(**supplied)


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_dicts(child)


def test_artifact_mismatch_withholds_every_aggregate_rate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact_calls = {"count": 0}

    def changing_artifacts() -> dict[str, dict[str, str]]:
        artifact_calls["count"] += 1
        result = fake_artifact_hashes()

        if artifact_calls["count"] >= 2:
            result["runner"]["sha256"] = "c" * 64

        return result

    def matching_receipt(kwargs: dict[str, Any]) -> dict[str, Any]:
        receipt = make_valid_repetition(
            seconds=2.0,
            fingerprint="same-source",
        )
        receipt["phase"] = kwargs.get("phase", "measured")
        receipt["repetition_index"] = kwargs.get("repetition_index", 0)
        return receipt

    result = _invoke_synthetic_run_benchmark(
        monkeypatch,
        tmp_path,
        receipt_factory=matching_receipt,
        artifact_factory=changing_artifacts,
    )

    assert artifact_calls["count"] >= 2
    assert result["valid"] is False
    assert result["status"] == "invalid"

    error_text = " ".join(_nested_strings(result)).lower()
    assert "artifact" in error_text
    assert any(
        word in error_text
        for word in ("changed", "mismatch", "different", "hash")
    )

    _assert_aggregate_rates_withheld(result)
    json.dumps(result, allow_nan=False)


# ---------------------------------------------------------------------------
# CLI defaults and validation
# ---------------------------------------------------------------------------

def test_cli_defaults_are_one_warmup_and_five_measured_repetitions() -> None:
    parser = a22._build_parser()
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest is not None
    }

    warmup_destinations = [
        name
        for name in defaults
        if "warmup" in name.lower()
    ]
    measured_destinations = [
        name
        for name in defaults
        if "repetition" in name.lower() and "warmup" not in name.lower()
    ]

    assert warmup_destinations
    assert measured_destinations
    assert any(defaults[name] == 1 for name in warmup_destinations)
    assert any(defaults[name] == 5 for name in measured_destinations)


def test_cli_defaults_use_seed_seven_and_single_world_single_candidate() -> None:
    parser = a22._build_parser()
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest is not None
    }

    assert defaults.get("seed") == 7
    assert defaults.get("batch") == 1
    assert defaults.get("candidates") == 1


def test_validate_arguments_rejects_nonpositive_dimensions() -> None:
    signature = inspect.signature(a22._validate_arguments)
    candidate_values = {
        "seed": 7,
        "batch": 0,
        "candidates": 1,
        "warmups": 1,
        "repetitions": 5,
        "warmup_repetitions": 1,
        "measured_repetitions": 5,
    }
    supplied = {
        name: candidate_values[name]
        for name in signature.parameters
        if name in candidate_values
    }

    if "batch" in signature.parameters:
        with pytest.raises((ValueError, a22.A22InvalidRun)):
            a22._validate_arguments(**supplied)


# ---------------------------------------------------------------------------
# One real, unmocked production smoke test
# ---------------------------------------------------------------------------

def test_real_cpu_seed7_b1_n1_full_2400_step_episode_smoke(
    tmp_path: Path,
) -> None:
    """Persist the raw production receipt, then assert its evidence fields."""
    root = Path(__file__).resolve().parents[1]

    # This is deliberately unmocked: real A1 reset, controller, public step,
    # source identity, frozen-P0 comparison, and artifact hashing are used.
    receipt = a22.execute_repetition(
        root=root,
        seed=7,
        batch=1,
        candidates=1,
        repetition_index=0,
        phase="measured",
    )

    # Check the real function signature before using its documented
    # (receipt, path) argument order.
    write_signature = inspect.signature(a22.write_receipt)
    write_parameters = list(write_signature.parameters)

    assert len(write_parameters) >= 2
    assert write_parameters[0] == "receipt"
    assert write_parameters[1] in ("path", "output_path")

    receipt_path = tmp_path / "a22_seed7_b1_n1_raw_receipt.json"
    a22.write_receipt(receipt, receipt_path)

    assert receipt_path.is_file()
    persisted_receipt = json.loads(
        receipt_path.read_text(encoding="utf-8")
    )
    assert persisted_receipt == receipt

    assert receipt["phase"] == "measured"
    assert receipt["seed"] == 7
    assert receipt["batch"] == 1
    assert receipt["candidates"] == 1
    assert receipt["device"] == "cpu"
    assert receipt["valid"] is True
    assert receipt["status"] == "valid"

    assert receipt["transitions_attempted"] == 2400
    assert receipt["public_step_calls_returned"] == 2400

    assert (
        isinstance(receipt["episode_seconds"], (int, float))
        and math.isfinite(receipt["episode_seconds"])
        and receipt["episode_seconds"] > 0.0
    )
    assert (
        isinstance(receipt["episode_steps_per_second"], (int, float))
        and math.isfinite(receipt["episode_steps_per_second"])
        and receipt["episode_steps_per_second"] > 0.0
    )
    assert (
        isinstance(receipt["world_steps_per_second"], (int, float))
        and math.isfinite(receipt["world_steps_per_second"])
        and receipt["world_steps_per_second"] > 0.0
    )

    before = receipt["source_identity_before"]
    after = receipt["source_identity_after"]

    assert before["identity_complete"] is True
    assert after["identity_complete"] is True
    assert before["revision"] == after["revision"]
    assert (
        receipt["source_fingerprint_before"]
        == receipt["source_fingerprint_after"]
    )

    preservation = before["frozen_p0_preservation"]
    assert preservation["status"] == "verified"
    assert preservation["file_count"] == 20
    assert preservation["compared_file_count"] == 20
    assert preservation["mismatched_files"] == []

    assert receipt["artifact_hashes"]
    assert all(
        artifact["status"] == "verified"
        for artifact in receipt["artifact_hashes"].values()
    )

    assert receipt["final_state"]
    assert receipt["final_state"]["geometry_unchanged"] is True
    assert receipt["final_state"]["final_geometry"] == (
        receipt["initial_state"]["geometry"]
    )

    json.dumps(receipt, allow_nan=False)


def _invoke_synthetic_run_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    receipt_factory: Any,
    artifact_factory: Any,
) -> Any:
    """Run the actual aggregate path with synthetic repetition receipts."""
    signature = inspect.signature(a22.run_benchmark)

    candidate_values = {
        "root": tmp_path,
        "seed": 7,
        "batch": 1,
        "candidates": 1,
        "warmups": 1,
        "warmup_repetitions": 1,
        "repetitions": 2,
        "measured_repetitions": 2,
        "output": None,
        "output_path": None,
        "device": "cpu",
    }

    supplied = {}
    for name, parameter in signature.parameters.items():
        if name in candidate_values:
            supplied[name] = candidate_values[name]
        elif parameter.default is inspect.Parameter.empty:
            raise AssertionError(
                "Unmapped required run_benchmark parameter: "
                f"{name!r}"
            )

    monkeypatch.setattr(
        a22,
        "execute_repetition",
        lambda **kwargs: receipt_factory(kwargs),
    )
    monkeypatch.setattr(
        a22,
        "_runtime_metadata",
        lambda *_args, **_kwargs: {
            "device": "cpu",
            "synthetic": True,
        },
    )
    monkeypatch.setattr(
        a22,
        "_artifact_hashes",
        lambda _root: artifact_factory(),
    )

    return a22.run_benchmark(**supplied)


def _nested_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _nested_dicts(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _nested_dicts(child)


def _nested_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _nested_strings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _nested_strings(child)


def _assert_aggregate_rates_withheld(result: Any) -> None:
    aggregate_rate_fields = []
    aggregate_duration_fields = []

    aggregate_markers = (
        "median",
        "mean",
        "min",
        "max",
        "range",
        "spread",
        "std",
    )

    for item in _nested_dicts(result):
        for key, value in item.items():
            lowered = key.lower()
            is_aggregate = any(
                marker in lowered for marker in aggregate_markers
            )

            if not is_aggregate:
                continue

            if (
                "steps_per_second" in lowered
                or "world_steps_per_second" in lowered
            ):
                aggregate_rate_fields.append((key, value))

            if "episode_seconds" in lowered:
                aggregate_duration_fields.append((key, value))

    assert aggregate_rate_fields, (
        "No aggregate throughput fields were found in the result"
    )
    assert aggregate_duration_fields, (
        "No aggregate episode-duration fields were found in the result"
    )

    assert all(value is None for _, value in aggregate_rate_fields)
    assert all(value is None for _, value in aggregate_duration_fields)


@pytest.mark.parametrize(
    ("interval", "timer_values", "expected_field"),
    [
        (
            "setup",
            [2.0, 1.0],
            "setup_seconds",
        ),
        (
            "setup",
            [float("nan"), 2.0],
            "setup_seconds",
        ),
        (
            "setup",
            [1.0, float("inf")],
            "setup_seconds",
        ),
        (
            "reset",
            [1.0, 2.0, 4.0, 3.0],
            "reset_seconds",
        ),
        (
            "reset",
            [1.0, 2.0, 3.0, float("nan")],
            "reset_seconds",
        ),
        (
            "reset",
            [1.0, 2.0, 3.0, float("inf")],
            "reset_seconds",
        ),
    ],
)
def test_negative_or_nonfinite_setup_and_reset_intervals_are_rejected_json_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interval: str,
    timer_values: list[float],
    expected_field: str,
) -> None:
    install_synthetic_execute_dependencies(
        monkeypatch,
        loop_steps=1,
        timer_values=timer_values,
    )

    receipt = run_synthetic_repetition(tmp_path)

    assert receipt["valid"] is False
    assert receipt[expected_field] is None
    assert receipt["episode_seconds"] is None
    assert receipt["episode_steps_per_second"] is None
    assert receipt["world_steps_per_second"] is None
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize(
    "failed_interval_end",
    [
        float("nan"),
        float("inf"),
        -1.0,
    ],
)
def test_nonfinite_or_negative_interrupted_interval_is_strict_json_safe(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failed_interval_end: float,
) -> None:
    install_synthetic_execute_dependencies(
        monkeypatch,
        loop_steps=3,
        step_failure_at=1,
        timer_values=[
            1.0,
            1.0,
            2.0,
            2.0,
            3.0,
            failed_interval_end,
        ],
    )

    monkeypatch.setattr(a21, "per_world_steps", lambda _env: [0])
    monkeypatch.setattr(
        a21,
        "per_world_completed",
        lambda _env: [False],
    )
    monkeypatch.setattr(
        a21,
        "final_inventory",
        lambda _env: {"partial": True},
    )

    receipt = run_synthetic_repetition(tmp_path)

    assert receipt["valid"] is False
    assert receipt["transitions_attempted"] is None
    assert receipt["public_step_calls_returned"] is None
    assert receipt["failed_interval_elapsed_seconds"] is None
    json.dumps(receipt, allow_nan=False)


def test_execute_repetition_rejects_changed_before_after_source_and_keeps_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install_synthetic_execute_dependencies(
        monkeypatch,
        loop_steps=3,
    )

    identities = iter(
        [
            fake_source_identity("source-before"),
            fake_source_identity("source-after"),
        ]
    )
    monkeypatch.setattr(
        a22,
        "_get_source_identity",
        lambda _root: next(identities),
    )

    receipt = run_synthetic_repetition(tmp_path)

    assert receipt["valid"] is False
    assert receipt["status"] == "invalid"
    assert receipt["transitions_attempted"] == 3
    assert receipt["public_step_calls_returned"] == 3
    assert receipt["episode_seconds"] is None
    assert receipt["episode_steps_per_second"] is None
    assert receipt["world_steps_per_second"] is None
    assert "source identity changed" in receipt["error"].lower()
    json.dumps(receipt, allow_nan=False)


def test_run_benchmark_rejects_warmup_measured_source_identity_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def phase_dependent_receipt(kwargs: dict[str, Any]) -> dict[str, Any]:
        phase = kwargs.get("phase", "measured")
        fingerprint = (
            "warmup-source"
            if phase == "warmup"
            else "measured-source"
        )
        receipt = make_valid_repetition(
            seconds=2.0,
            fingerprint=fingerprint,
        )
        receipt["phase"] = phase
        receipt["repetition_index"] = kwargs.get("repetition_index", 0)
        return receipt

    result = _invoke_synthetic_run_benchmark(
        monkeypatch,
        tmp_path,
        receipt_factory=phase_dependent_receipt,
        artifact_factory=fake_artifact_hashes,
    )

    assert result["valid"] is False
    assert result["status"] == "invalid"

    error_text = " ".join(_nested_strings(result)).lower()
    assert "source" in error_text
    assert any(
        word in error_text
        for word in ("consistent", "changed", "fingerprint", "identity")
    )

    _assert_aggregate_rates_withheld(result)
    json.dumps(result, allow_nan=False)


def _synthetic_final_validator_call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_geometry: Any = None,
    initial_geometry: Any = None,
    step_counts: list[int] | None = None,
    completed: list[bool] | None = None,
    inventory: dict[str, Any] | None = None,
    invalid_physical_state: bool = False,
) -> dict[str, Any]:
    """Build a valid synthetic shifted world, then optionally vary one defect."""
    batch = 1
    candidates = 1
    config = SimpleNamespace()

    full_steps = a22.A21_EPISODE_STEPS

    if initial_geometry is None:
        initial_geometry = {"synthetic_layout": "unchanged"}

    if final_geometry is None:
        final_geometry = initial_geometry

    if step_counts is None:
        step_counts = [full_steps]

    if completed is None:
        completed = [True]

    if inventory is None:
        inventory = valid_final_inventory(
            batch=batch,
            candidates=candidates,
        )

    # The synthetic world has completed the configured signed 90-degree
    # gravity shift: initial [4, 0] -> final [0, 4], direction +1.
    environment = make_physical_environment(
        batch=batch,
        candidates=candidates,
    )
    world = environment.worlds[0]
    world.gravity = torch.tensor([0.0, 4.0], dtype=torch.float32)
    world.shift_applied = True
    world.shift_direction = 1

    environment.config = config
    environment.num_agents = candidates
    environment.num_worlds = batch
    environment.all_completed = all(completed)

    if invalid_physical_state:
        world.episode.physics.alive[0, 0] = False

    # Include both the independently queried state and the corresponding
    # final-inventory fields. These baseline values agree.
    inventory = dict(inventory)
    inventory.setdefault("world_step_counts", list(step_counts))
    inventory.setdefault("world_completed", list(completed))

    initial_gravity = [[4.0, 0.0]]
    final_gravity = [[0.0, 4.0]]

    monkeypatch.setattr(
        a21,
        "per_world_steps",
        lambda _environment: list(step_counts),
    )
    monkeypatch.setattr(
        a21,
        "per_world_completed",
        lambda _environment: list(completed),
    )
    monkeypatch.setattr(
        a21,
        "geometry_inventory",
        lambda _environment: final_geometry,
    )
    monkeypatch.setattr(
        a21,
        "validate_a21_geometry",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        a21,
        "a21_gravity_snapshot",
        lambda _environment: final_gravity,
    )
    monkeypatch.setattr(
        a21,
        "final_inventory",
        lambda _environment: inventory,
    )

    observations = torch.zeros(
        (batch, candidates, OBSERVATION_WIDTH),
        dtype=torch.float32,
        device="cpu",
    )

    signature = inspect.signature(a22._validate_final_environment)
    values = {
        "environment": environment,
        "batch": batch,
        "candidates": candidates,
        "config": config,
        "initial_gravity": initial_gravity,
        "initial_geometry": initial_geometry,
        "final_observations": observations,
    }

    kwargs = {}
    for name, parameter in signature.parameters.items():
        if name in values:
            kwargs[name] = values[name]
        elif parameter.default is inspect.Parameter.empty:
            raise AssertionError(
                "Unmapped required _validate_final_environment parameter: "
                f"{name!r}"
            )

    result = a22._validate_final_environment(**kwargs)

    return {
        "result": result,
        "environment": environment,
        "inventory": inventory,
        "initial_geometry": initial_geometry,
        "final_geometry": final_geometry,
    }


def test_synthetic_final_validator_default_state_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic = _synthetic_final_validator_call(monkeypatch)

    assert synthetic["result"]
    assert synthetic["result"]["geometry_unchanged"] is True
    assert synthetic["result"]["final_geometry"] == (
        synthetic["initial_geometry"]
    )

    world = synthetic["environment"].worlds[0]
    assert world.shift_applied is True
    assert world.shift_direction == 1
    assert torch.equal(
        world.gravity,
        torch.tensor([0.0, 4.0], dtype=torch.float32),
    )

    assert synthetic["inventory"]["world_step_counts"] == [
        a22.A21_EPISODE_STEPS
    ]
    assert synthetic["inventory"]["world_completed"] == [True]


def test_final_environment_rejects_changed_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        a22.A22InvalidRun,
        match="(?i)geometry",
    ):
        _synthetic_final_validator_call(
            monkeypatch,
            initial_geometry={"layout": "before"},
            final_geometry={"layout": "after"},
        )


def test_final_environment_rejects_invalid_final_physical_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(
        a22.A22InvalidRun,
        match="(?i)(alive|dead|physical)",
    ):
        _synthetic_final_validator_call(
            monkeypatch,
            invalid_physical_state=True,
        )


@pytest.mark.parametrize(
    ("step_counts", "completed", "inventory_overrides", "match"),
    [
        (
            [a22.A21_EPISODE_STEPS - 1],
            [True],
            {"world_step_counts": [a22.A21_EPISODE_STEPS - 1],
             "world_completed": [True]},
            r"(?i)(advance|2400|transitions)",
        ),
        (
            [a22.A21_EPISODE_STEPS],
            [False],
            {"world_step_counts": [a22.A21_EPISODE_STEPS],
             "world_completed": [False]},
            r"(?i)(completed|completion)",
        ),
        (
            [a22.A21_EPISODE_STEPS],
            [True],
            {"world_step_counts": [a22.A21_EPISODE_STEPS],
             "world_completed": [True],
             "truncated_per_world": [0]},
            r"(?i)(truncated|deadline)",
        ),
    ],
)
def test_final_environment_rejects_bad_completion_or_deadline_inventory(
    monkeypatch,
    step_counts,
    completed,
    inventory_overrides,
    match,
):
    inventory = valid_final_inventory(batch=1, candidates=1)
    inventory.update(inventory_overrides)

    with pytest.raises(a22.A22InvalidRun, match=match):
        _synthetic_final_validator_call(
            monkeypatch,
            step_counts=step_counts,
            completed=completed,
            inventory=inventory,
        )
