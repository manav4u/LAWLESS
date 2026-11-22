"""Focused regression tests for LAWLESS A1 benchmark harness."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lawless.environment.reset import reset
from lawless.environment.step import step
from lawless.world.config import EnvironmentConfig

from benchmarks import environment_benchmark as bench


# ---------------------------------------------------------------------------
# Real reset/step integration
# ---------------------------------------------------------------------------

def make_real_config(seed: int) -> EnvironmentConfig:
    config = EnvironmentConfig()
    config.seed = seed
    config.validate()
    return config


def test_real_b1_n4_bounded_smoke():
    """Exercise actual production reset and step for B=1, N=4."""
    config = make_real_config(7)

    environment, observations, reset_info = reset(
        config,
        1,
        4,
        device="cpu",
        dtype=torch.float32,
    )

    assert environment.num_worlds == 1
    assert environment.num_agents == 4
    assert len(environment.worlds) == 1

    before = bench.pre_step_counts(environment)
    assert before["worlds_advancing"] == 1
    assert before["alive_agent_transitions_expected"] == 4

    actions = torch.zeros((1, 4, 2), dtype=torch.float32)

    (
        next_environment,
        next_observations,
        rewards,
        terminated,
        truncated,
        info,
    ) = step(environment, actions)

    accounting = bench.empty_accounting(1, 4)
    bench.add_step_accounting(
        accounting,
        before,
        bench.per_world_completed(environment),
        next_environment,
        info,
    )

    assert accounting["step_calls"] == 1
    assert accounting["world_step_transitions"] == 1
    assert accounting["candidate_slot_transitions"] == 4
    assert accounting["alive_agent_transitions"] == 4
    assert accounting["per_world_step_transitions"] == [1]
    assert accounting["per_world_candidate_slot_transitions"] == [4]

    assert len(next_environment.worlds) == 1
    assert isinstance(info, dict)


def test_seed_is_applied_and_world_seed_changes():
    """Verify actual root seed propagation through canonical reset."""
    config7 = make_real_config(7)
    config8 = make_real_config(8)

    env7, _, _ = reset(
        config7,
        1,
        4,
        device="cpu",
        dtype=torch.float32,
    )

    env8, _, _ = reset(
        config8,
        1,
        4,
        device="cpu",
        dtype=torch.float32,
    )

    seeds7 = bench.per_world_seeds(env7)
    seeds8 = bench.per_world_seeds(env8)

    assert config7.seed == 7
    assert config8.seed == 8
    assert seeds7 != seeds8


# ---------------------------------------------------------------------------
# Accounting helper regression tests
# ---------------------------------------------------------------------------

def make_fake_environment(
    worlds,
    num_agents=1,
):
    return SimpleNamespace(
        worlds=worlds,
        num_agents=num_agents,
        num_worlds=len(worlds),
    )


def make_fake_world(
    *,
    completed=False,
    alive=None,
    extracted=None,
    failed=None,
    truncated=None,
    step_count=0,
):
    if alive is None:
        alive = torch.tensor([True])

    if extracted is None:
        extracted = torch.tensor([False])

    if failed is None:
        failed = torch.tensor([False])

    if truncated is None:
        truncated = torch.tensor([False])

    episode = SimpleNamespace(
        physics=SimpleNamespace(alive=alive),
        embodiment=SimpleNamespace(
            extracted=extracted,
            failed=failed,
        ),
    )

    return SimpleNamespace(
        completed=completed,
        truncated=truncated,
        episode=episode,
        step_count=step_count,
    )


def test_alive_before_transition_counts_agent_that_dies():
    """Alive-at-start counts the death transition."""
    world = make_fake_world(
        alive=torch.tensor([True]),
        step_count=0,
    )
    environment = make_fake_environment([world], num_agents=1)

    before = bench.pre_step_counts(environment)
    assert before["alive_agent_transitions_expected"] == 1

    # Simulate post-transition death.
    world.episode.physics.alive[:] = False
    world.episode.embodiment.failed[:] = True
    world.step_count = 1

    accounting = bench.empty_accounting(1, 1)
    bench.add_step_accounting(
        accounting,
        before,
        [False],
        environment,
        {
            "failure_this_step": torch.tensor([True]),
            "energy_death_this_step": torch.tensor([True]),
        },
    )

    assert accounting["alive_agent_transitions"] == 1
    assert accounting["per_world_alive_transitions"] == [1]
    assert accounting["transition_event_counts"][
        "failure_this_step"
    ] == 1
    assert accounting["transition_event_counts"][
        "energy_death_this_step"
    ] == 1


def test_frozen_timeout_world_excluded_even_if_physically_alive():
    """Completed/truncated frozen world does not count live physical mask."""
    frozen = make_fake_world(
        completed=True,
        alive=torch.tensor([True]),
        truncated=torch.tensor([True]),
        step_count=20,
    )
    active = make_fake_world(
        completed=False,
        alive=torch.tensor([True]),
        step_count=5,
    )

    environment = make_fake_environment(
        [frozen, active],
        num_agents=1,
    )

    before = bench.pre_step_counts(environment)

    assert before["advancing"] == [False, True]
    assert before["alive_at_transition_start"] == [0, 1]
    assert before["worlds_advancing"] == 1
    assert before["alive_agent_transitions_expected"] == 1

    accounting = bench.empty_accounting(2, 1)

    active.step_count += 1

    bench.add_step_accounting(
        accounting,
        before,
        [True, False],
        environment,
        {},
    )

    assert accounting["world_step_transitions"] == 1
    assert accounting["candidate_slot_transitions"] == 1
    assert accounting["alive_agent_transitions"] == 1
    assert accounting["per_world_step_transitions"] == [0, 1]
    assert accounting["per_world_alive_transitions"] == [0, 1]
    assert accounting["per_world_candidate_slot_transitions"] == [0, 1]


def test_newly_completed_worlds_not_persistent_masks():
    """Completion event is counted once, not for persistent completion."""
    world = make_fake_world(
        completed=False,
        alive=torch.tensor([True]),
        step_count=0,
    )
    environment = make_fake_environment([world], num_agents=1)
    accounting = bench.empty_accounting(1, 1)

    before = bench.pre_step_counts(environment)
    previous = [False]

    world.completed = True
    world.step_count = 1

    bench.add_step_accounting(
        accounting,
        before,
        previous,
        environment,
        {},
    )

    assert accounting["newly_completed_worlds"] == 1
    assert accounting["completion_events"] == [
        {"world_index": 0, "step_index": 1}
    ]

    # A subsequent invocation should not step this completed world.
    before2 = bench.pre_step_counts(environment)
    assert before2["worlds_advancing"] == 0
    assert before2["alive_agent_transitions_expected"] == 0

    bench.add_step_accounting(
        accounting,
        before2,
        [True],
        environment,
        {},
    )

    assert accounting["newly_completed_worlds"] == 1
    assert len(accounting["completion_events"]) == 1


def test_event_count_keys_are_exact_and_transition_only():
    info = {
        "newly_collected": torch.tensor([True, False]),
        "newly_extracted": torch.tensor([False, True]),
        "failure_this_step": torch.tensor([False, False]),
        "energy_death_this_step": torch.tensor([False, True]),
        "collision_contact": torch.tensor([True, True]),
        "unresolved_contact": torch.tensor([False, True]),
        "boundary_escape": torch.tensor([False, False]),
        "shift_applied_this_step": torch.tensor([True]),
        # Persistent state deliberately not used as event.
        "world_completed": torch.tensor([True, True]),
        "truncated": torch.tensor([True, True]),
    }

    counts = bench.event_counts(info)

    assert set(counts) == set(bench.EVENT_KEYS)
    assert counts == {
        "newly_collected": 1,
        "newly_extracted": 1,
        "failure_this_step": 0,
        "energy_death_this_step": 1,
        "collision_contact": 2,
        "unresolved_contact": 1,
        "boundary_escape": 0,
        "shift_applied_this_step": 1,
    }


# ---------------------------------------------------------------------------
# Source identity tests
# ---------------------------------------------------------------------------

def test_source_identity_uses_dynamic_inventory(monkeypatch, tmp_path):
    """Inventory derives from Git, not hardcoded nonexistent paths."""
    monkeypatch.setattr(
        bench,
        "tracked_python_inventory",
        lambda root: [
            "lawless/world/full_generator.py",
            "lawless/sensors/observations.py",
        ],
    )

    monkeypatch.setattr(
        bench,
        "untracked_python_inventory",
        lambda root: ["benchmarks/environment_benchmark.py"],
    )

    monkeypatch.setattr(
        bench,
        "git_dirty_status",
        lambda root: {
            "status": "verified",
            "dirty": True,
            "error": None,
        },
    )

    monkeypatch.setattr(
        bench,
        "git_success",
        lambda root, *args: "test_revision",
    )

    monkeypatch.setattr(
        bench,
        "file_hash_inventory",
        lambda root, paths: {
            "status": "verified",
            "entries": {
                name: {"status": "verified", "sha256": "abc"}
                for name in paths
            },
            "missing_or_error": [],
        },
    )

    monkeypatch.setattr(
        bench,
        "frozen_tag_python_inventory",
        lambda root: {
            "status": "verified",
            "tag": bench.FROZEN_P0_TAG,
            "entries": {},
            "missing_or_error": [],
        },
    )

    identity = bench.source_identity(tmp_path)

    assert identity["status"] == "verified"
    assert identity["tracked_python_files"] == [
        "lawless/world/full_generator.py",
        "lawless/sensors/observations.py",
    ]
    assert identity["untracked_python_files"] == [
        "benchmarks/environment_benchmark.py"
    ]
    assert identity["dirty_status"]["dirty"] is True
    assert identity["identity_complete"] is True


def test_source_identity_failure_is_not_verified(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bench,
        "tracked_python_inventory",
        lambda root: None,
    )
    monkeypatch.setattr(
        bench,
        "untracked_python_inventory",
        lambda root: None,
    )
    monkeypatch.setattr(
        bench,
        "git_dirty_status",
        lambda root: {
            "status": "unknown",
            "dirty": None,
            "error": "not a repository",
        },
    )
    monkeypatch.setattr(
        bench,
        "git_success",
        lambda root, *args: None,
    )
    monkeypatch.setattr(
        bench,
        "file_hash_inventory",
        lambda root, paths: {
            "status": "incomplete",
            "entries": {},
            "missing_or_error": list(paths),
        },
    )
    monkeypatch.setattr(
        bench,
        "frozen_tag_python_inventory",
        lambda root: {
            "status": "unknown",
            "tag": bench.FROZEN_P0_TAG,
            "entries": {},
            "missing_or_error": [],
        },
    )

    identity = bench.source_identity(tmp_path)

    assert identity["status"] == "incomplete"
    assert identity["identity_complete"] is False
    assert identity["dirty_status"]["dirty"] is None


def test_frozen_identity_failure_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bench,
        "git_command",
        lambda root, *args: None,
    )

    identity = bench.frozen_tag_python_inventory(tmp_path)

    assert identity["status"] == "unknown"
    assert identity["entries"] == {}


# ---------------------------------------------------------------------------
# Checkpoint, failure and output-preservation tests
# ---------------------------------------------------------------------------

def test_atomic_checkpoint_preserves_prior_file_on_replace_failure(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "receipt.json"
    path.write_text('{"old":true}\n', encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("replace failure")

    monkeypatch.setattr(bench.os, "replace", fail_replace)

    with pytest.raises(OSError):
        bench.atomic_write_json(path, {"new": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "old": True
    }


def test_checkpoint_status_is_failure_when_step_raises(
    monkeypatch,
    tmp_path,
):
    """Actual reset is used; the step entry point is fault-injected."""
    config = make_real_config(7)

    environment, observations, reset_info = reset(
        config,
        1,
        4,
        device="cpu",
        dtype=torch.float32,
    )

    # Patch the module's reset and step to avoid another reset while
    # exercising the run-level failure receipt. Reset returns a real state.
    monkeypatch.setattr(
        bench,
        "reset",
        lambda *args, **kwargs: (
            environment,
            observations,
            reset_info,
        ),
    )

    def fail_step(*args, **kwargs):
        raise RuntimeError("injected step failure")

    monkeypatch.setattr(bench, "step", fail_step)
    monkeypatch.setattr(
        bench,
        "source_identity",
        lambda root: {
            "status": "verified",
            "identity_complete": True,
        },
    )

    output = tmp_path / "run.json"

    run = bench.execute_run(
        root=tmp_path,
        mode="test",
        batch=1,
        candidates=4,
        seed=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_steps=2,
        post_reset_wall_budget=10.0,
        output_path=output,
    )

    assert run["status"] == "failed"
    assert "injected step failure" in run["exception"]
    assert output.is_file()

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "traceback" in receipt


def test_invocation_output_directory_is_not_reused(tmp_path):
    """CLI invocation design is unique; this checks output file isolation."""
    first = tmp_path / "invocation_1"
    second = tmp_path / "invocation_2"

    first.mkdir()
    second.mkdir()

    receipt1 = first / "run_000.json"
    receipt2 = second / "run_000.json"

    bench.atomic_write_json(receipt1, {"seed": 1})
    bench.atomic_write_json(receipt2, {"seed": 2})

    assert json.loads(receipt1.read_text())["seed"] == 1
    assert json.loads(receipt2.read_text())["seed"] == 2


# ---------------------------------------------------------------------------
# Timing and budget validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "budget",
    [0, -1, -0.5],
)
def test_nonpositive_wall_budget_rejected(budget):
    with pytest.raises(ValueError):
        bench.validate_wall_budget(budget)


def test_positive_wall_budget_validated():
    bench.validate_wall_budget(0.001)


@pytest.mark.parametrize(
    "steps",
    [0, -1],
)
def test_nonpositive_step_bound_rejected(steps):
    with pytest.raises(ValueError):
        bench.validate_step_bound(steps)


def test_positive_step_bound_validated():
    bench.validate_step_bound(2)


def test_rate_scope_is_accounted_not_uninstrumented():
    run = {
        "status": "bounded_incomplete",
        "termination_reason": "max_steps_reached",
        "elapsed_post_reset_wall_seconds": 2.0,
        "reset_wall_seconds": 1.0,
        "step_call_samples_seconds": [0.2, 0.3],
        "accounting": {
            "alive_agent_transitions": 10,
            "candidate_slot_transitions": 20,
            "world_step_transitions": 2,
        },
        "final_inventory": {},
    }

    result = bench.run_summary(run)

    assert (
        result["accounted_rates"]["rate_scope"]
        == "accounted_bounded_A1"
    )
    assert (
        result["accounted_rates"][
            "alive_agent_transitions_per_post_reset_wall_second"
        ]
        == 5
    )
    assert (
        result["accounted_step_call_samples"]["count"]
        == 2
    )