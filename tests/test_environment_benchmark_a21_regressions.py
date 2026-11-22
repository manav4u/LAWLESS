from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from benchmarks import environment_benchmark as bench


REPO_ROOT = Path(__file__).resolve().parents[1]


def _reset(seed: int, batch: int, candidates: int):
    config = bench.create_a21_config(seed)
    return bench.reset(
        config,
        batch,
        candidates,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _shifted_case(
    *,
    reported_gravity: torch.Tensor | None = None,
    installed_gravity: torch.Tensor | None = None,
    reported_persistent: bool = True,
    actual_persistent: bool = True,
):
    """Build a shift-transition info/state pair with independently settable claims."""
    environment, observations, reset_info = _reset(
        seed=7,
        batch=1,
        candidates=1,
    )
    world = environment.worlds[0]

    gravity_used = world.gravity.detach().clone()
    direction = world.shift_direction
    expected_gravity = torch.stack(
        (-direction * gravity_used[1], direction * gravity_used[0])
    )

    if reported_gravity is None:
        reported_gravity = expected_gravity
    if installed_gravity is None:
        installed_gravity = expected_gravity

    shifted_world = replace(
        world,
        gravity=installed_gravity.to(
            device=world.gravity.device,
            dtype=world.gravity.dtype,
        ),
        step_count=bench.A21_GRAVITY_SHIFT_STEP,
        shift_applied=actual_persistent,
    )
    shifted_environment = replace(
        environment,
        worlds=(shifted_world,),
    )

    info = {
        "shift_applied_this_step": torch.tensor([True]),
        "shift_applied": torch.tensor([reported_persistent]),
        "elapsed_steps": torch.tensor(
            [bench.A21_GRAVITY_SHIFT_STEP],
            dtype=torch.int64,
        ),
        "elapsed_time_s": torch.tensor(
            [bench.A21_SHIFT_TIME_S],
            dtype=torch.float64,
        ),
        "gravity_used": gravity_used.reshape(1, 2),
        "gravity_next": reported_gravity.reshape(1, 2),
    }

    return shifted_environment, info, gravity_used, expected_gravity


def test_a21_zero_action_policy_retains_seed7_boundary_escape_regression():
    """The original failing policy remains a reproducible negative case."""
    environment, observations, reset_info = _reset(
        seed=7,
        batch=1,
        candidates=1,
    )

    zero_actions = torch.zeros(
        (1, 1, 2),
        dtype=torch.float32,
    )

    boundary_escape_seen = False
    failure_seen = False
    actual_transition = None

    for transition in range(1, 319):
        (
            next_environment,
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = bench.step(environment, zero_actions)

        environment = next_environment

        boundary_escape = info["boundary_escape"]
        failure_this_step = info["failure_this_step"]

        if bool(boundary_escape.any().item()):
            boundary_escape_seen = True
            actual_transition = transition

        if bool(failure_this_step.any().item()):
            failure_seen = True

        if boundary_escape_seen:
            break

    assert boundary_escape_seen
    assert failure_seen
    assert actual_transition == 318


def test_a21_shift_validator_rejects_missing_shift_event():
    # This intentionally limited dictionary verifies that the missing-event
    # check runs before the validator needs an environment or other info.
    info = {
        "shift_applied_this_step": torch.tensor([False]),
    }

    with pytest.raises(
        RuntimeError,
        match="gravity-shift event did not match",
    ):
        bench.a21_validate_shift_transition(
            info,
            environment=object(),
            transition=bench.A21_GRAVITY_SHIFT_STEP,
        )


def test_a21_shift_validator_rejects_reported_but_ineffective_shift():
    """A true event flag cannot substitute for a changed gravity vector."""
    environment, observations, reset_info = _reset(
        seed=7,
        batch=1,
        candidates=1,
    )

    gravity = environment.worlds[0].gravity.detach().clone()

    # Deliberately keep the vector unchanged. This test isolates the
    # ineffective-vector condition before counter/state validation.
    info = {
        "shift_applied_this_step": torch.tensor([True]),
        "shift_applied": torch.tensor([True]),
        "gravity_used": gravity.reshape(1, 2),
        "gravity_next": gravity.reshape(1, 2).clone(),
        "elapsed_time_s": torch.tensor(
            [bench.A21_SHIFT_TIME_S],
            dtype=torch.float64,
        ),
    }

    with pytest.raises(
        RuntimeError,
        match="did not change the applied gravity vector",
    ):
        bench.a21_validate_shift_transition(
            info,
            environment,
            bench.A21_GRAVITY_SHIFT_STEP,
        )


def test_a21_shift_validator_accepts_actual_installed_quarter_turn():
    environment, info, gravity_used, expected_gravity = _shifted_case()

    observation = bench.a21_validate_shift_transition(
        info,
        environment,
        bench.A21_GRAVITY_SHIFT_STEP,
    )

    assert observation is not None
    assert observation["transition"] == bench.A21_GRAVITY_SHIFT_STEP
    assert observation["elapsed_time_s"] == pytest.approx(20.0)
    assert observation["shift_event_per_world"] == [1]
    assert observation["persistent_shift_applied_per_world"] == [True]
    assert observation["actual_world_shift_applied_per_world"] == [True]
    assert observation["actual_world_step_counts"] == [1200]
    assert observation["gravity_used_first_world"] == pytest.approx(
        gravity_used.cpu().tolist()
    )
    assert observation["gravity_next_first_world"] == pytest.approx(
        expected_gravity.cpu().tolist()
    )
    assert observation["post_shift_world_gravity_vectors"][0] == pytest.approx(
        expected_gravity.cpu().tolist()
    )


def test_a21_shift_validator_rejects_reported_but_uninstalled_gravity():
    environment, info, gravity_used, expected_gravity = _shifted_case(
        reported_gravity=None,
        installed_gravity=None,
    )

    # Keep the production persistent state and step counter consistent, but
    # deliberately replace the returned state's gravity with its pre-shift
    # vector. The info report still claims the correct quarter-turn.
    world = environment.worlds[0]
    unshifted_world = replace(world, gravity=gravity_used.clone())
    environment = replace(environment, worlds=(unshifted_world,))

    assert torch.allclose(info["gravity_next"][0], expected_gravity)

    with pytest.raises(RuntimeError, match="was not installed"):
        bench.a21_validate_shift_transition(
            info,
            environment,
            bench.A21_GRAVITY_SHIFT_STEP,
        )


@pytest.mark.parametrize("rotation_error", ["wrong_direction", "wrong_angle"])
def test_a21_shift_validator_rejects_wrong_rotation(rotation_error):
    environment, info, gravity_used, expected_gravity = _shifted_case()

    direction = environment.worlds[0].shift_direction

    if rotation_error == "wrong_direction":
        wrong_gravity = torch.stack(
            (direction * gravity_used[1], -direction * gravity_used[0])
        )
    else:
        # Same magnitude, but a 180-degree rotation rather than a quarter-turn.
        wrong_gravity = -gravity_used

    wrong_world = replace(
        environment.worlds[0],
        gravity=wrong_gravity.clone(),
    )
    wrong_environment = replace(
        environment,
        worlds=(wrong_world,),
    )
    wrong_info = dict(info)
    wrong_info["gravity_next"] = wrong_gravity.reshape(1, 2)

    assert torch.allclose(
        torch.linalg.vector_norm(wrong_info["gravity_next"], dim=-1),
        torch.linalg.vector_norm(wrong_info["gravity_used"], dim=-1),
    )
    assert not torch.allclose(wrong_gravity, expected_gravity)

    with pytest.raises(
        RuntimeError,
        match="signed 90-degree rotation",
    ):
        bench.a21_validate_shift_transition(
            wrong_info,
            wrong_environment,
            bench.A21_GRAVITY_SHIFT_STEP,
        )


def test_a21_shift_validator_rejects_persistent_info_state_not_installed():
    environment, info, gravity_used, expected_gravity = _shifted_case(
        reported_persistent=True,
        actual_persistent=False,
    )

    assert torch.allclose(info["gravity_next"][0], expected_gravity)
    assert torch.allclose(
        environment.worlds[0].gravity,
        expected_gravity,
    )

    with pytest.raises(
        RuntimeError,
        match="actual world shift_applied state",
    ):
        bench.a21_validate_shift_transition(
            info,
            environment,
            bench.A21_GRAVITY_SHIFT_STEP,
        )


def test_a21_frozen_world_is_excluded_from_transition_accounting():
    """Use production step with one already-completed world and one active."""
    environment, observations, reset_info = _reset(
        seed=7,
        batch=2,
        candidates=2,
    )

    frozen_world = replace(
        environment.worlds[0],
        truncated=torch.ones_like(environment.worlds[0].truncated),
        completed=True,
    )

    environment = replace(
        environment,
        worlds=(frozen_world, environment.worlds[1]),
    )

    previous_completed = bench.per_world_completed(environment)
    before = bench.pre_step_counts(environment)

    assert before["advancing"] == [False, True]
    assert before["worlds_advancing"] == 1
    assert before["alive_at_transition_start"][0] == 0

    actions = torch.zeros(
        (2, 2, 2),
        dtype=torch.float32,
    )

    next_environment, observations, rewards, terminated, truncated, info = (
        bench.step(environment, actions)
    )

    accounting = bench.empty_accounting(batch=2, candidates=2)

    bench.add_step_accounting(
        accounting,
        before,
        previous_completed,
        next_environment,
        info,
    )

    assert bench.per_world_steps(next_environment) == [0, 1]
    assert accounting["step_calls"] == 1
    assert accounting["world_step_transitions"] == 1
    assert accounting["candidate_slot_transitions"] == 2
    assert accounting["per_world_step_transitions"] == [0, 1]
    assert accounting["per_world_candidate_slot_transitions"] == [0, 2]

    # A completed world may retain a true physical alive mask because
    # completion can be due to truncation. It must remain excluded.
    assert bool(
        next_environment.worlds[0].episode.physics.alive.any().item()
    )
    assert next_environment.worlds[0].completed


def test_a21_early_completion_is_rejected_and_receipt_keeps_returned_state(
    tmp_path,
    monkeypatch,
):
    """The receipt must reflect the returned state before invariant failure."""
    original_step = bench.step
    injected = {"done": False}

    def step_with_early_completion(environment, actions):
        result = original_step(environment, actions)

        (
            next_environment,
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = result

        if not injected["done"]:
            worlds = list(next_environment.worlds)
            worlds[0] = replace(
                worlds[0],
                truncated=torch.ones_like(worlds[0].truncated),
                completed=True,
            )
            next_environment = replace(
                next_environment,
                worlds=tuple(worlds),
            )
            injected["done"] = True

        return (
            next_environment,
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )

    monkeypatch.setattr(bench, "step", step_with_early_completion)

    output = tmp_path / "a21_early_completion.json"

    result = bench.execute_a21_stress_fixture(
        root=REPO_ROOT,
        batch=1,
        candidates=1,
        seed=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
        output_path=output,
    )

    assert result["status"] == "failed"
    assert "completed before the deadline" in result["exception"]

    assert result["partial_inventory"]["world_step_counts"] == [1]
    assert result["partial_inventory"]["world_completed"] == [True]

    assert result["accounting"]["step_calls"] == 1
    assert result["accounting"]["world_step_transitions"] == 1
    assert result["accounting"]["candidate_slot_transitions"] == 1
    assert result["accounting"]["newly_completed_worlds"] == 1

    assert output.is_file()


def test_a21_receipt_keeps_last_returned_state_if_next_invariant_fails(
    tmp_path,
    monkeypatch,
):
    """A post-step invariant failure must not leave a stale receipt."""
    original_step = bench.step

    def step_with_nonfinite_observation(environment, actions):
        (
            next_environment,
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = original_step(environment, actions)

        observations = observations.clone()
        observations.reshape(-1)[0] = float("nan")

        return (
            next_environment,
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )

    monkeypatch.setattr(
        bench,
        "step",
        step_with_nonfinite_observation,
    )

    output = tmp_path / "a21_nonfinite_observation.json"

    result = bench.execute_a21_stress_fixture(
        root=REPO_ROOT,
        batch=1,
        candidates=1,
        seed=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
        output_path=output,
    )

    assert result["status"] == "failed"
    assert "observations contain non-finite values" in result["exception"]

    assert result["partial_inventory"]["world_step_counts"] == [1]
    assert result["accounting"]["step_calls"] == 1
    assert result["accounting"]["world_step_transitions"] == 1
    assert output.is_file()


def test_a21_fixture_rejects_transient_geometry_change_after_accounting(
    tmp_path,
    monkeypatch,
):
    """A transition-2 geometry defect is caught and fully accounted."""
    original_geometry_inventory = bench.geometry_inventory
    inventory_calls = {"count": 0}

    def transient_geometry_inventory(environment):
        inventory_calls["count"] += 1

        # First call: reset inventory.
        # Second call: transition 1 inventory.
        # Third call: transient invalid inventory at transition 2.
        if inventory_calls["count"] == 3:
            return [{"transient_invalid_geometry": True}]

        return original_geometry_inventory(environment)

    monkeypatch.setattr(
        bench,
        "geometry_inventory",
        transient_geometry_inventory,
    )

    output = tmp_path / "a21_transient_geometry.json"

    result = bench.execute_a21_stress_fixture(
        root=REPO_ROOT,
        batch=1,
        candidates=1,
        seed=7,
        device=torch.device("cpu"),
        dtype=torch.float32,
        output_path=output,
    )

    assert result["status"] == "failed"
    assert "geometry inventory changed during the episode" in result["exception"]
    assert "transition 2" in result["exception"]

    # The second returned state was installed before accounting and geometry
    # validation. Both physical state and accounting must reach transition 2.
    assert result["partial_inventory"]["world_step_counts"] == [2]
    assert result["accounting"]["step_calls"] == 2
    assert result["accounting"]["world_step_transitions"] == 2
    assert result["accounting"]["candidate_slot_transitions"] == 2

    assert result["geometry_validation_checks"] == 2
    assert result["geometry_world_validation_checks"] == 2
    assert result["geometry_last_checked_transition"] == 2
    assert result["geometry_validation_complete"] is False

    assert inventory_calls["count"] == 3
    assert output.is_file()

    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"] == "failed"
    assert saved["partial_inventory"]["world_step_counts"] == [2]
    assert saved["accounting"]["step_calls"] == 2
    assert saved["geometry_validation_checks"] == 2