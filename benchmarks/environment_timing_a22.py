"""A2.2 uninstrumented CPU timing runner for the locked A2.1 episode.

Run from the repository root, for example:

    python -m benchmarks.environment_timing_a22

The default workload is seed 7, batch 1, candidates 1, float32 CPU,
one full warm-up episode, and five measured full episodes.

This module does not modify production or A2.1 code. It times the existing
A2.1 controller and production public step path. All correctness, source
identity, and final-state checks run outside the timed episode interval.

A receipt is evidence from this runner only. It does not establish overall
P1.8-A acceptance, authorize a main merge, or constitute a P6 release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from benchmarks import environment_benchmark as a21


A22_RUNNER_VERSION = 1
A22_DEFAULT_SEED = 7
A22_DEFAULT_BATCH = 1
A22_DEFAULT_CANDIDATES = 1
A22_DEFAULT_WARMUPS = 1
A22_DEFAULT_REPETITIONS = 5

A21_EPISODE_STEPS = 2400
A21_GRAVITY_SHIFT_STEP = 1200
A21_EPISODE_DURATION_S = 40.0
A21_DT_S = 1.0 / 60.0
A21_GRAVITY_M_S2 = 4.0

DEVICE = torch.device("cpu")
DTYPE = torch.float32


class A22InvalidRun(RuntimeError):
    """Raised when a timing repetition cannot be accepted as valid evidence."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    """Serialize common runtime values without silently discarding evidence."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return vars(value)
    return repr(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
        allow_nan=False,
    )


def _sha256_file(path: Path, *, root: Path) -> dict[str, Any]:
    """Hash a file while recording its repository-relative path."""
    try:
        relative_path = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return {
            "path": str(path),
            "status": "outside_repository",
            "sha256": None,
            "size_bytes": None,
        }

    if not path.is_file():
        return {
            "path": relative_path,
            "status": "missing",
            "sha256": None,
            "size_bytes": None,
        }

    digest = hashlib.sha256()
    size = 0

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)

    return {
        "path": relative_path,
        "status": "verified",
        "sha256": digest.hexdigest(),
        "size_bytes": size,
    }


def _artifact_hashes(root: Path) -> dict[str, Any]:
    """Hash the A1 benchmark, A2.2 runner, and focused A2.2 tests."""
    root = root.resolve()
    paths = {
        "a21_benchmark": root / "benchmarks" / "environment_benchmark.py",
        "a22_runner": root / "benchmarks" / "environment_timing_a22.py",
        "a22_tests": root / "tests" / "test_environment_benchmark_a22.py",
    }
    return {
        name: _sha256_file(path, root=root)
        for name, path in paths.items()
    }


def _identity_fingerprint(identity: dict[str, Any]) -> str:
    """Fingerprint dynamic source identity, excluding no identity fields."""
    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def _hash_entries(inventory: Any, *, label: str) -> dict[str, str]:
    """Normalize a source-inventory entries mapping to path -> SHA-256."""
    if not isinstance(inventory, dict):
        raise A22InvalidRun(f"{label} hash inventory is not a dictionary")

    entries = inventory.get("entries")
    if not isinstance(entries, dict):
        raise A22InvalidRun(f"{label} hash inventory has no entries mapping")

    normalized: dict[str, str] = {}

    for path, value in entries.items():
        if not isinstance(path, str) or not path:
            raise A22InvalidRun(
                f"{label} hash inventory contains an invalid path"
            )

        if isinstance(value, str):
            digest = value
        elif isinstance(value, dict):
            digest = value.get("sha256")
        else:
            digest = None

        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise A22InvalidRun(
                f"{label} hash for {path!r} is missing or malformed"
            )

        normalized[path] = digest

    return normalized


def _finite_seconds(value: Any, *, label: str, positive: bool) -> float:
    """Validate a measured interval and return its finite float value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise A22InvalidRun(f"{label} is not a numeric timing interval")

    result = float(value)

    if not math.isfinite(result):
        raise A22InvalidRun(f"{label} is not finite")

    if positive and result <= 0.0:
        raise A22InvalidRun(f"{label} must be strictly positive")

    if not positive and result < 0.0:
        raise A22InvalidRun(f"{label} must not be negative")

    return result


def _safe_elapsed(start: Any, end: Any) -> float | None:
    """Return a finite elapsed interval, or None for unusable timer edges."""
    try:
        start_value = float(start)
        end_value = float(end)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(start_value) or not math.isfinite(end_value):
        return None

    elapsed = end_value - start_value
    if not math.isfinite(elapsed) or elapsed < 0.0:
        return None

    return elapsed


def _common_source_fingerprint(repetitions: list[dict[str, Any]]) -> str:
    """Require every repetition to use the same verified dynamic identity."""
    fingerprints: list[str] = []

    for repetition in repetitions:
        before = repetition.get("source_fingerprint_before")
        after = repetition.get("source_fingerprint_after")

        if not isinstance(before, str) or not isinstance(after, str):
            raise A22InvalidRun(
                "a repetition is missing its before/after source fingerprint"
            )

        if before != after:
            raise A22InvalidRun(
                "a repetition's dynamic source identity changed during execution"
            )

        fingerprints.append(before)

    if not fingerprints:
        raise A22InvalidRun("no source identities are available to compare")

    if len(set(fingerprints)) != 1:
        raise A22InvalidRun(
            "warm-up and measured repetitions do not share one actual "
            "source identity"
        )

    return fingerprints[0]


def _partial_progress(environment: Any) -> dict[str, Any]:
    """Capture actual world progress without guessing public-call counts."""
    progress: dict[str, Any] = {
        "transitions_attempted": None,
        "public_step_calls_returned": None,
        "world_step_counts": None,
        "minimum_world_step_count": None,
        "world_completed": None,
        "all_completed": None,
    }

    if environment is None:
        return progress

    try:
        steps = a21.per_world_steps(environment)
        progress["world_step_counts"] = steps

        if steps:
            progress["minimum_world_step_count"] = min(steps)

        progress["world_completed"] = a21.per_world_completed(environment)
        progress["all_completed"] = bool(environment.all_completed)

        try:
            progress["partial_final_inventory"] = a21.final_inventory(
                environment
            )
        except Exception as exc:
            progress["partial_inventory_error"] = (
                f"{type(exc).__name__}: {exc}"
            )

    except Exception as exc:
        progress["progress_snapshot_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    # Per-world counters are state evidence, not an exact count of calls
    # returned by the public step function. Leave that count unknown here.
    return progress


def _validate_final_physical_inventory(
    environment: Any,
    inventory: dict[str, Any],
    *,
    batch: int,
    candidates: int,
) -> None:
    """Require finite physical state and deadline-truncated live candidates."""
    expected_counts = {
        "physical_alive_per_world": candidates,
        "extracted_per_world": 0,
        "failed_per_world": 0,
        "truncated_per_world": candidates,
    }

    for field, expected in expected_counts.items():
        values = inventory.get(field)

        if not isinstance(values, list) or len(values) != batch:
            raise A22InvalidRun(
                f"final inventory field {field!r} must contain one count "
                "per world"
            )

        for world_index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, int):
                raise A22InvalidRun(
                    f"{field}[{world_index}] is not an integer count"
                )

            if value != expected:
                raise A22InvalidRun(
                    f"{field}[{world_index}]={value}; expected {expected}. "
                    "The controlled full-duration workload requires every "
                    "candidate to remain alive and be truncated at deadline, "
                    "with no extraction or failure."
                )

    for world_index, world in enumerate(environment.worlds):
        physics = world.episode.physics

        # Use the production state object's own shape validator first.
        validate_shapes = getattr(physics, "validate_shapes", None)
        if not callable(validate_shapes):
            raise A22InvalidRun(
                f"world {world_index} physics state has no validate_shapes()"
            )

        try:
            validate_shapes()
        except Exception as exc:
            raise A22InvalidRun(
                f"world {world_index} physics shape validation failed: {exc}"
            ) from exc

        expected_shapes = {
            "position": (candidates, 2),
            "velocity": (candidates, 2),
            "angle": (candidates, 1),
            "angular_velocity": (candidates, 1),
            "energy": (candidates, 1),
            "alive": (candidates, 1),
        }

        for field_name, expected_shape in expected_shapes.items():
            value = getattr(physics, field_name, None)

            if not isinstance(value, torch.Tensor):
                raise A22InvalidRun(
                    f"world {world_index} physics field {field_name!r} "
                    "is not a tensor"
                )

            if tuple(value.shape) != expected_shape:
                raise A22InvalidRun(
                    f"world {world_index} physics field {field_name!r} "
                    f"has shape {tuple(value.shape)}; "
                    f"expected {expected_shape}"
                )

            if value.device.type != "cpu":
                raise A22InvalidRun(
                    f"world {world_index} physics field {field_name!r} "
                    f"is on {value.device}; expected CPU"
                )

            if field_name == "alive":
                if value.dtype != torch.bool:
                    raise A22InvalidRun(
                        f"world {world_index} alive dtype is {value.dtype}; "
                        "expected torch.bool"
                    )
            elif not value.is_floating_point():
                raise A22InvalidRun(
                    f"world {world_index} physics field {field_name!r} "
                    f"has non-floating dtype {value.dtype}"
                )

            if value.numel() == 0 or not bool(
                torch.isfinite(value).all().item()
            ):
                raise A22InvalidRun(
                    f"world {world_index} physics field {field_name!r} "
                    "is empty or non-finite"
                )

        if not bool(physics.alive.all().item()):
            raise A22InvalidRun(
                f"world {world_index} contains dead candidates at deadline"
            )

        gravity = getattr(world, "gravity", None)
        if not isinstance(gravity, torch.Tensor):
            raise A22InvalidRun(
                f"world {world_index} has no tensor-valued gravity state"
            )

        if gravity.device.type != "cpu":
            raise A22InvalidRun(
                f"world {world_index} gravity is on {gravity.device}; "
                "expected CPU"
            )

        if gravity.numel() == 0 or not bool(
            torch.isfinite(gravity).all().item()
        ):
            raise A22InvalidRun(
                f"world {world_index} gravity is empty or non-finite"
            )


def _get_source_identity(root: Path) -> dict[str, Any]:
    identity = a21.source_identity(root)

    if not isinstance(identity, dict):
        raise A22InvalidRun("A1 source_identity did not return a dictionary")

    if identity.get("status") != "verified":
        raise A22InvalidRun(
            "dynamic source identity is incomplete: "
            f"status={identity.get('status')!r}"
        )

    if identity.get("identity_complete") is not True:
        raise A22InvalidRun("dynamic source identity is not complete")

    if identity.get("benchmark_included") is not True:
        raise A22InvalidRun(
            "dynamic source identity does not include the A1 benchmark"
        )

    if not identity.get("revision"):
        raise A22InvalidRun("the checked-out git revision is unavailable")

    current_inventory = identity.get("current_python_hashes")
    frozen_inventory = identity.get("frozen_p0_python_hashes")

    if not isinstance(current_inventory, dict):
        raise A22InvalidRun("current Python source hashes are missing")
    if current_inventory.get("status") != "verified":
        raise A22InvalidRun("current Python source hashes are incomplete")

    if not isinstance(frozen_inventory, dict):
        raise A22InvalidRun("frozen P0 source hashes are missing")
    if frozen_inventory.get("status") != "verified":
        raise A22InvalidRun("frozen P0 source hashes are incomplete")

    current_entries = _hash_entries(
        current_inventory,
        label="current Python source",
    )
    frozen_entries = _hash_entries(
        frozen_inventory,
        label="frozen P0",
    )

    if len(frozen_entries) != 20:
        raise A22InvalidRun(
            "frozen P0 inventory must contain exactly 20 Python files; "
            f"received {len(frozen_entries)}"
        )

    missing = sorted(set(frozen_entries) - set(current_entries))
    if missing:
        raise A22InvalidRun(
            "current source inventory is missing frozen P0 Python files: "
            + ", ".join(missing)
        )

    changed = sorted(
        path
        for path, frozen_digest in frozen_entries.items()
        if current_entries[path] != frozen_digest
    )
    if changed:
        raise A22InvalidRun(
            "frozen P0 Python source differs from its frozen-tag hash: "
            + ", ".join(changed)
        )

    identity = dict(identity)
    identity["frozen_p0_preservation"] = {
        "status": "verified",
        "file_count": len(frozen_entries),
        "compared_file_count": len(frozen_entries),
        "mismatched_files": [],
    }

    return identity


def _runtime_metadata() -> dict[str, Any]:
    metadata = a21.runtime_metadata(DEVICE)
    metadata.update(
        {
            "timestamp_utc": _utc_now(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": str(DEVICE),
            "dtype": str(DTYPE),
            "torch_num_threads": int(torch.get_num_threads()),
            "torch_num_interop_threads": int(torch.get_num_interop_threads()),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        }
    )
    return metadata


def _validate_arguments(
    *,
    batch: int,
    candidates: int,
    seed: int,
    warmups: int,
    repetitions: int,
) -> None:
    a21.validate_positive_integer(batch, "batch")
    a21.validate_positive_integer(candidates, "candidates")
    a21.validate_positive_integer(warmups, "warmups")
    a21.validate_positive_integer(repetitions, "repetitions")
    a21.validate_seed_span(seed, 1)

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _tensor_is_finite(value: Any) -> bool:
    if not isinstance(value, torch.Tensor):
        return False
    if value.numel() == 0:
        return False
    return bool(torch.isfinite(value).all().item())


def _validate_observations(
    observations: Any,
    *,
    batch: int,
    candidates: int,
    label: str,
) -> None:
    expected_shape = (batch, candidates, 23)

    if not isinstance(observations, torch.Tensor):
        raise A22InvalidRun(f"{label} observations are not a tensor")

    if tuple(observations.shape) != expected_shape:
        raise A22InvalidRun(
            f"{label} observations have shape {tuple(observations.shape)}; "
            f"expected the A2.1 CPU observation contract {expected_shape}"
        )

    if observations.device.type != "cpu":
        raise A22InvalidRun(
            f"{label} observations are on {observations.device}; expected CPU"
        )

    if observations.dtype != torch.float32:
        raise A22InvalidRun(
            f"{label} observations have dtype {observations.dtype}; "
            "expected torch.float32"
        )

    if observations.numel() == 0 or not bool(
        torch.isfinite(observations).all().item()
    ):
        raise A22InvalidRun(
            f"{label} observations are empty or contain non-finite values"
        )


def _capture_target_positions(environment: Any) -> torch.Tensor:
    if not hasattr(environment, "worlds") or not environment.worlds:
        raise A22InvalidRun("reset returned an environment with no worlds")

    try:
        return torch.stack(
            [
                world.episode.physics.position
                for world in environment.worlds
            ],
            dim=0,
        ).detach().clone()
    except Exception as exc:
        raise A22InvalidRun(
            f"could not capture controller target positions: {exc}"
        ) from exc


def _validate_initial_environment(
    environment: Any,
    *,
    batch: int,
    candidates: int,
    config: Any,
) -> dict[str, Any]:
    if len(environment.worlds) != batch:
        raise A22InvalidRun(
            f"reset returned {len(environment.worlds)} worlds; expected {batch}"
        )

    if int(environment.num_agents) != candidates:
        raise A22InvalidRun(
            f"reset returned {environment.num_agents} candidates; "
            f"expected {candidates}"
        )

    if environment.config is not config:
        raise A22InvalidRun(
            "reset environment does not retain the exact A2.1 config object"
        )

    if bool(environment.all_completed):
        raise A22InvalidRun("all worlds are already completed after reset")

    initial_steps = a21.per_world_steps(environment)
    if initial_steps != [0] * batch:
        raise A22InvalidRun(
            f"worlds did not start at step zero: {initial_steps!r}"
        )

    initial_completed = a21.per_world_completed(environment)
    if initial_completed != [False] * batch:
        raise A22InvalidRun(
            f"worlds were completed immediately after reset: "
            f"{initial_completed!r}"
        )

    geometry = a21.geometry_inventory(environment)
    a21.validate_a21_geometry(geometry, batch)

    gravity = a21.a21_gravity_snapshot(environment)
    if len(gravity) != batch:
        raise A22InvalidRun(
            f"expected initial gravity for {batch} worlds, received "
            f"{len(gravity)}"
        )

    for index, vector in enumerate(gravity):
        if len(vector) != 2 or not all(math.isfinite(float(x)) for x in vector):
            raise A22InvalidRun(
                f"world {index} has invalid initial gravity vector {vector!r}"
            )

        magnitude = math.hypot(float(vector[0]), float(vector[1]))
        if not math.isclose(
            magnitude,
            A21_GRAVITY_M_S2,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise A22InvalidRun(
                f"world {index} initial gravity magnitude is {magnitude}; "
                f"expected {A21_GRAVITY_M_S2}"
            )

    return {
        "geometry": geometry,
        "initial_gravity": gravity,
        "initial_world_steps": initial_steps,
        "initial_world_completed": initial_completed,
        "world_seeds": a21.per_world_seeds(environment),
    }


def _validate_final_environment(
    environment: Any,
    *,
    batch: int,
    candidates: int,
    config: Any,
    initial_gravity: list[list[float]],
    initial_geometry: list[dict[str, Any]],
    final_observations: Any,
) -> dict[str, Any]:
    if len(environment.worlds) != batch:
        raise A22InvalidRun(
            f"final environment has {len(environment.worlds)} worlds; "
            f"expected {batch}"
        )

    if int(environment.num_agents) != candidates:
        raise A22InvalidRun(
            f"final environment has {environment.num_agents} candidates; "
            f"expected {candidates}"
        )

    if environment.config is not config:
        raise A22InvalidRun(
            "final environment no longer retains the exact A2.1 config object"
        )

    world_steps = a21.per_world_steps(environment)
    if world_steps != [A21_EPISODE_STEPS] * batch:
        raise A22InvalidRun(
            "one or more worlds did not advance on all 2400 transitions: "
            f"{world_steps!r}"
        )

    completed = a21.per_world_completed(environment)
    if completed != [True] * batch:
        raise A22InvalidRun(
            f"not every world completed at the deadline: {completed!r}"
        )

    if not bool(environment.all_completed):
        raise A22InvalidRun(
            "environment all_completed is false after the deadline"
        )

    _validate_observations(
        final_observations,
        batch=batch,
        candidates=candidates,
        label="final",
    )

    final_geometry = a21.geometry_inventory(environment)
    a21.validate_a21_geometry(final_geometry, batch)

    if _canonical_json(final_geometry) != _canonical_json(initial_geometry):
        raise A22InvalidRun(
            "generated-world geometry changed during the timed episode"
        )

    final_gravity = a21.a21_gravity_snapshot(environment)
    if len(final_gravity) != batch:
        raise A22InvalidRun(
            f"expected final gravity for {batch} worlds, received "
            f"{len(final_gravity)}"
        )

    expected_gravity: list[list[float]] = []

    for index, world in enumerate(environment.worlds):
        if not hasattr(world, "shift_applied"):
            raise A22InvalidRun(
                f"world {index} has no persistent shift_applied state"
            )

        if not bool(world.shift_applied):
            raise A22InvalidRun(
                f"world {index} did not retain its gravity shift"
            )

        if not hasattr(world, "shift_direction"):
            raise A22InvalidRun(
                f"world {index} has no shift_direction state"
            )

        direction = int(world.shift_direction)
        if direction not in (-1, 1):
            raise A22InvalidRun(
                f"world {index} has invalid shift_direction {direction!r}"
            )

        gx, gy = (float(x) for x in initial_gravity[index])
        expected = [-direction * gy, direction * gx]
        expected_gravity.append(expected)

        actual = [float(x) for x in final_gravity[index]]

        if len(actual) != 2 or not all(math.isfinite(x) for x in actual):
            raise A22InvalidRun(
                f"world {index} has invalid final gravity {actual!r}"
            )

        if not all(
            math.isclose(
                actual_component,
                expected_component,
                rel_tol=0.0,
                abs_tol=1e-5,
            )
            for actual_component, expected_component
            in zip(actual, expected)
        ):
            raise A22InvalidRun(
                f"world {index} final gravity {actual!r} does not match "
                f"the expected persistent shift {expected!r}"
            )

        magnitude = math.hypot(actual[0], actual[1])
        if not math.isclose(
            magnitude,
            A21_GRAVITY_M_S2,
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise A22InvalidRun(
                f"world {index} final gravity magnitude is {magnitude}; "
                f"expected {A21_GRAVITY_M_S2}"
            )

    inventory = a21.final_inventory(environment)

    if inventory.get("world_step_counts") != [A21_EPISODE_STEPS] * batch:
        raise A22InvalidRun(
            "final inventory step counts do not confirm 2400 transitions"
        )

    if inventory.get("world_completed") != [True] * batch:
        raise A22InvalidRun(
            "final inventory does not confirm completion of every world"
        )

    _validate_final_physical_inventory(
        environment,
        inventory,
        batch=batch,
        candidates=candidates,
    )

    return {
        "final_inventory": inventory,
        "final_geometry": final_geometry,
        "geometry_unchanged": True,
        "final_gravity": final_gravity,
        "expected_final_gravity": expected_gravity,
        "final_observation_shape": list(final_observations.shape),
        "final_observations_finite": True,
        "final_observations_device": str(final_observations.device),
        "final_observations_dtype": str(final_observations.dtype),
        "final_physical_state_finite": True,
        "deadline_inventory_valid": True,
    }


def _validate_config(config: Any) -> None:
    config.validate()

    episode = config.episode
    expected_values = (
        ("episode_duration_s", episode.episode_duration_s,
         A21_EPISODE_DURATION_S),
        ("dt_s", episode.dt_s, A21_DT_S),
        ("gravity_m_s2", episode.gravity_m_s2, A21_GRAVITY_M_S2),
    )

    for name, actual, expected in expected_values:
        if not math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise A22InvalidRun(
                f"configuration episode.{name}={actual!r}; "
                f"expected {expected!r}"
            )

    computed_steps = int(round(
        float(episode.episode_duration_s) / float(episode.dt_s)
    ))
    if computed_steps != A21_EPISODE_STEPS:
        raise A22InvalidRun(
            f"configuration yields {computed_steps} steps; "
            f"expected {A21_EPISODE_STEPS}"
        )

    shift_step = int(round(
        float(episode.shift_time_s) / float(episode.dt_s)
    ))
    if shift_step != A21_GRAVITY_SHIFT_STEP:
        raise A22InvalidRun(
            f"configuration shifts at transition {shift_step}; "
            f"expected {A21_GRAVITY_SHIFT_STEP}"
        )


def execute_repetition(
    *,
    root: Path,
    seed: int,
    batch: int,
    candidates: int,
    repetition_index: int,
    phase: str,
) -> dict[str, Any]:
    """Execute one full A2.1 episode and retain its raw timing receipt."""
    if phase not in ("warmup", "measured"):
        raise ValueError("phase must be 'warmup' or 'measured'")

    root = root.resolve()

    receipt: dict[str, Any] = {
        "phase": phase,
        "repetition_index": repetition_index,
        "seed": seed,
        "batch": batch,
        "candidates": candidates,
        "device": str(DEVICE),
        "dtype": str(DTYPE),
        "status": "invalid",
        "valid": False,
        "setup_seconds": None,
        "reset_seconds": None,
        "episode_seconds": None,
        "unvalidated_episode_seconds": None,
        "failed_interval_elapsed_seconds": None,
        "transitions_attempted": None,
        "public_step_calls_returned": None,
        "episode_steps_per_second": None,
        "world_steps_per_second": None,
        "error": None,
        "traceback": None,
        "validation": {},
    }

    environment = None
    final_observations = None
    source_before = None
    source_after = None

    setup_start = None
    setup_end = None
    reset_start = None
    reset_end = None
    episode_start = None
    episode_end = None

    # These become exact only after the entire fixed loop returns normally.
    # They are not incremented inside the timed loop.
    full_loop_returned_normally = False

    try:
        source_before = _get_source_identity(root)
        receipt["source_identity_before"] = source_before
        receipt["source_fingerprint_before"] = _identity_fingerprint(
            source_before
        )

        setup_start = time.perf_counter()
        config = a21.create_a21_config(seed)
        _validate_config(config)
        setup_end = time.perf_counter()

        receipt["setup_seconds"] = _finite_seconds(
            _safe_elapsed(setup_start, setup_end),
            label="setup_seconds",
            positive=False,
        )

        reset_start = time.perf_counter()
        reset_result = a21.reset(
            config,
            batch,
            candidates,
            device=DEVICE,
            dtype=DTYPE,
        )
        reset_end = time.perf_counter()

        receipt["reset_seconds"] = _finite_seconds(
            _safe_elapsed(reset_start, reset_end),
            label="reset_seconds",
            positive=False,
        )

        if not isinstance(reset_result, tuple) or len(reset_result) != 3:
            raise A22InvalidRun(
                "A1 reset must return "
                "(environment, observations, reset_info)"
            )

        environment, reset_observations, reset_info = reset_result

        _validate_observations(
            reset_observations,
            batch=batch,
            candidates=candidates,
            label="reset",
        )

        initial_state = _validate_initial_environment(
            environment,
            batch=batch,
            candidates=candidates,
            config=config,
        )
        receipt["initial_state"] = initial_state
        receipt["reset_info"] = a21.json_safe(reset_info)

        target_positions = _capture_target_positions(environment)

        if tuple(target_positions.shape[:2]) != (batch, candidates):
            raise A22InvalidRun(
                "captured target positions do not match batch/candidate shape: "
                f"{tuple(target_positions.shape)}"
            )

        # Only the controller and production public step execute between
        # these timer edges. No per-transition counters or diagnostics.
        episode_start = time.perf_counter()

        for _transition in range(A21_EPISODE_STEPS):
            actions = a21.a21_controller_actions(
                environment,
                target_positions,
            )
            (
                environment,
                final_observations,
                _rewards,
                _terminated,
                _truncated,
                _info,
            ) = a21.step(environment, actions)

        episode_end = time.perf_counter()
        full_loop_returned_normally = True

        # The exact call/transition counts are now known: the fixed loop
        # returned from all 2400 public step calls. Preserve them even if a
        # subsequent correctness, timing, or source check rejects the run.
        receipt["transitions_attempted"] = A21_EPISODE_STEPS
        receipt["public_step_calls_returned"] = A21_EPISODE_STEPS

        raw_episode_seconds = _safe_elapsed(episode_start, episode_end)
        receipt["unvalidated_episode_seconds"] = raw_episode_seconds

        raw_episode_seconds = _finite_seconds(
            raw_episode_seconds,
            label="episode_seconds",
            positive=True,
        )

        final_state = _validate_final_environment(
            environment,
            batch=batch,
            candidates=candidates,
            config=config,
            initial_gravity=initial_state["initial_gravity"],
            initial_geometry=initial_state["geometry"],
            final_observations=final_observations,
        )
        receipt["final_state"] = final_state

        source_after = _get_source_identity(root)
        receipt["source_identity_after"] = source_after
        receipt["source_fingerprint_after"] = _identity_fingerprint(
            source_after
        )

        if receipt["source_fingerprint_before"] != (
            receipt["source_fingerprint_after"]
        ):
            raise A22InvalidRun(
                "dynamic source identity changed during the repetition"
            )

        receipt["artifact_hashes"] = _artifact_hashes(root)

        missing_artifacts = [
            name
            for name, item in receipt["artifact_hashes"].items()
            if item["status"] != "verified"
        ]
        if missing_artifacts:
            raise A22InvalidRun(
                "required benchmark/test artifact hashes are unavailable: "
                + ", ".join(missing_artifacts)
            )

        # Publish accepted duration and rates only after every repetition-
        # level timing, final-state, and source/artifact gate has passed.
        receipt["episode_seconds"] = raw_episode_seconds
        receipt["episode_steps_per_second"] = (
            A21_EPISODE_STEPS / raw_episode_seconds
        )
        receipt["world_steps_per_second"] = (
            A21_EPISODE_STEPS * batch / raw_episode_seconds
        )

        for field in (
            "episode_seconds",
            "episode_steps_per_second",
            "world_steps_per_second",
        ):
            value = receipt[field]
            if (
                not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise A22InvalidRun(
                    f"{field} is non-finite or non-positive"
                )

        receipt["valid"] = True
        receipt["status"] = "valid"

    except Exception as exc:
        receipt["status"] = "invalid"
        receipt["valid"] = False
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()

        if episode_start is not None and episode_end is None:
            receipt["failed_interval_elapsed_seconds"] = _safe_elapsed(
                episode_start,
                time.perf_counter(),
            )

        if full_loop_returned_normally:
            # A later validation failure must not erase the exact count
            # already established by normal return from the fixed loop.
            receipt["transitions_attempted"] = A21_EPISODE_STEPS
            receipt["public_step_calls_returned"] = A21_EPISODE_STEPS
        else:
            # On interruption, exact Python public-call count is deliberately
            # unknown. Capture only actual per-world production counters.
            progress = _partial_progress(environment)
            receipt["partial_progress"] = progress
            receipt["transitions_attempted"] = None
            receipt["public_step_calls_returned"] = None

        if episode_end is not None:
            receipt["unvalidated_episode_seconds"] = _safe_elapsed(
                episode_start,
                episode_end,
            )

        if source_before is not None:
            receipt.setdefault("source_identity_before", source_before)
        if source_after is not None:
            receipt.setdefault("source_identity_after", source_after)

    # All raw intervals must be strict-JSON-safe. Setup/reset permit zero;
    # episode timing must be strictly positive to become accepted evidence.
    for field in (
        "setup_seconds",
        "reset_seconds",
        "episode_seconds",
        "unvalidated_episode_seconds",
        "failed_interval_elapsed_seconds",
        "episode_steps_per_second",
        "world_steps_per_second",
    ):
        value = receipt.get(field)
        if value is None:
            continue

        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            receipt[field] = None
            continue

        if not math.isfinite(numeric) or numeric < 0.0:
            receipt[field] = None

    if receipt["valid"] is not True:
        receipt["episode_seconds"] = None
        receipt["episode_steps_per_second"] = None
        receipt["world_steps_per_second"] = None

    return receipt


def _summarize_repetitions(
    repetitions: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [
        item
        for item in repetitions
        if item.get("valid") is True
        and isinstance(item.get("episode_seconds"), (int, float))
        and not isinstance(item.get("episode_seconds"), bool)
        and math.isfinite(float(item["episode_seconds"]))
        and float(item["episode_seconds"]) > 0.0
    ]

    summary: dict[str, Any] = {
        "measured_repetition_count": len(repetitions),
        "valid_repetition_count": len(valid),
        "invalid_repetition_count": len(repetitions) - len(valid),
        "all_measured_repetitions_valid": (
            len(repetitions) > 0 and len(valid) == len(repetitions)
        ),
        "episode_seconds": None,
        "median_episode_seconds": None,
        "minimum_episode_seconds": None,
        "maximum_episode_seconds": None,
        "spread_episode_seconds": None,
        "median_episode_steps_per_second": None,
        "median_world_steps_per_second": None,
    }

    if not repetitions or len(valid) != len(repetitions):
        return summary

    samples = [float(item["episode_seconds"]) for item in valid]
    if not all(math.isfinite(sample) and sample > 0.0 for sample in samples):
        return summary

    median_seconds = statistics.median(samples)
    minimum_seconds = min(samples)
    maximum_seconds = max(samples)
    spread_seconds = maximum_seconds - minimum_seconds

    if not all(
        math.isfinite(value)
        for value in (
            median_seconds,
            minimum_seconds,
            maximum_seconds,
            spread_seconds,
        )
    ):
        return summary

    if median_seconds <= 0.0:
        return summary

    batch_values = {int(item["batch"]) for item in valid}
    if len(batch_values) != 1:
        return summary

    batch = batch_values.pop()
    median_steps_rate = A21_EPISODE_STEPS / median_seconds
    median_world_rate = A21_EPISODE_STEPS * batch / median_seconds

    if not all(
        math.isfinite(value) and value > 0.0
        for value in (median_steps_rate, median_world_rate)
    ):
        return summary

    summary.update(
        {
            "episode_seconds": samples,
            "median_episode_seconds": median_seconds,
            "minimum_episode_seconds": minimum_seconds,
            "maximum_episode_seconds": maximum_seconds,
            "spread_episode_seconds": spread_seconds,
            "median_episode_steps_per_second": median_steps_rate,
            "median_world_steps_per_second": median_world_rate,
        }
    )
    return summary


def run_benchmark(
    *,
    root: Path,
    seed: int = A22_DEFAULT_SEED,
    batch: int = A22_DEFAULT_BATCH,
    candidates: int = A22_DEFAULT_CANDIDATES,
    warmups: int = A22_DEFAULT_WARMUPS,
    repetitions: int = A22_DEFAULT_REPETITIONS,
) -> dict[str, Any]:
    """Run warm-ups and measured repetitions and return the complete receipt."""
    root = root.resolve()

    _validate_arguments(
        batch=batch,
        candidates=candidates,
        seed=seed,
        warmups=warmups,
        repetitions=repetitions,
    )

    receipt: dict[str, Any] = {
        "schema": "lawless.a22.cpu_timing_receipt",
        "runner_version": A22_RUNNER_VERSION,
        "created_utc": _utc_now(),
        "status": "invalid",
        "valid": False,
        "scope": {
            "description": (
                "CPU uninstrumented timing of the A2.1 controller plus "
                "2400 production public step calls"
            ),
            "production_code_modified": False,
            "a21_code_modified": False,
            "frozen_p0_modified": False,
            "component_profiling_included": False,
            "main_merge_claimed": False,
            "overall_p18a_acceptance_claimed": False,
            "p6_release_claimed": False,
        },
        "workload": {
            "seed": seed,
            "batch": batch,
            "candidates": candidates,
            "device": str(DEVICE),
            "dtype": str(DTYPE),
            "episode_steps": A21_EPISODE_STEPS,
            "episode_duration_s": A21_EPISODE_DURATION_S,
            "dt_s": A21_DT_S,
            "gravity_shift_transition": A21_GRAVITY_SHIFT_STEP,
            "warmups": warmups,
            "measured_repetitions": repetitions,
        },
        "runtime": _runtime_metadata(),
        "artifact_hashes_before": _artifact_hashes(root),
        "warmup_repetitions": [],
        "measured_repetitions": [],
        "common_source_fingerprint": None,
        "summary": None,
        "error": None,
    }

    try:
        if any(
            item["status"] != "verified"
            for item in receipt["artifact_hashes_before"].values()
        ):
            raise A22InvalidRun(
                "one or more required runner/benchmark/test files could not "
                "be hashed before the run"
            )

        for index in range(warmups):
            warmup = execute_repetition(
                root=root,
                seed=seed,
                batch=batch,
                candidates=candidates,
                repetition_index=index,
                phase="warmup",
            )
            receipt["warmup_repetitions"].append(warmup)

            if warmup.get("valid") is not True:
                raise A22InvalidRun(
                    f"warm-up repetition {index} failed: "
                    f"{warmup.get('error')}"
                )

        for index in range(repetitions):
            measured = execute_repetition(
                root=root,
                seed=seed,
                batch=batch,
                candidates=candidates,
                repetition_index=index,
                phase="measured",
            )
            receipt["measured_repetitions"].append(measured)

        # All episodes, including warm-ups, must use one stable actual
        # revision and complete dynamic Python-source identity.
        all_repetitions = (
            receipt["warmup_repetitions"]
            + receipt["measured_repetitions"]
        )
        receipt["common_source_fingerprint"] = (
            _common_source_fingerprint(all_repetitions)
        )

        # Compare artifacts before calculating or publishing aggregate rates.
        receipt["artifact_hashes_after"] = _artifact_hashes(root)

        if _canonical_json(receipt["artifact_hashes_before"]) != (
            _canonical_json(receipt["artifact_hashes_after"])
        ):
            raise A22InvalidRun(
                "runner, A1 benchmark, or focused A2.2 test artifact changed "
                "during the benchmark run"
            )

        if not all(
            item.get("valid") is True
            for item in receipt["measured_repetitions"]
        ):
            raise A22InvalidRun(
                "one or more measured repetitions failed validation"
            )

        receipt["summary"] = _summarize_repetitions(
            receipt["measured_repetitions"]
        )

        if (
            receipt["summary"]["all_measured_repetitions_valid"] is not True
            or receipt["summary"]["median_episode_seconds"] is None
            or receipt["summary"]["median_episode_steps_per_second"] is None
            or receipt["summary"]["median_world_steps_per_second"] is None
        ):
            raise A22InvalidRun(
                "measured repetition summary failed finite timing validation"
            )

        receipt["status"] = "valid"
        receipt["valid"] = True

    except Exception as exc:
        receipt["status"] = "invalid"
        receipt["valid"] = False
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["traceback"] = traceback.format_exc()

        # An overall source/artifact failure invalidates the aggregate even
        # when individual repetitions previously passed their local checks.
        receipt["summary"] = {
            "measured_repetition_count": len(
                receipt["measured_repetitions"]
            ),
            "valid_repetition_count": sum(
                item.get("valid") is True
                for item in receipt["measured_repetitions"]
            ),
            "invalid_repetition_count": sum(
                item.get("valid") is not True
                for item in receipt["measured_repetitions"]
            ),
            "all_measured_repetitions_valid": False,
            "episode_seconds": None,
            "median_episode_seconds": None,
            "minimum_episode_seconds": None,
            "maximum_episode_seconds": None,
            "spread_episode_seconds": None,
            "median_episode_steps_per_second": None,
            "median_world_steps_per_second": None,
            "aggregate_withheld_reason": (
                "overall source/artifact/repetition gate did not pass"
            ),
        }

        receipt.setdefault("artifact_hashes_after", _artifact_hashes(root))

    if receipt.get("valid") is not True:
        summary = receipt.get("summary")
        if isinstance(summary, dict):
            for field in (
                "episode_seconds",
                "median_episode_seconds",
                "minimum_episode_seconds",
                "maximum_episode_seconds",
                "spread_episode_seconds",
                "median_episode_steps_per_second",
                "median_world_steps_per_second",
            ):
                summary[field] = None

    return receipt


def write_receipt(receipt: dict[str, Any], output_path: Path) -> None:
    """Write JSON atomically so interrupted writes do not leave partial JSON."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(
        receipt,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        default=_json_default,
        allow_nan=False,
    ) + "\n"

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the A2.2 uninstrumented CPU timing benchmark using the "
            "existing A2.1 controller and production public step."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository root (default: current working directory)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=A22_DEFAULT_SEED,
        help=f"fixed workload seed (default: {A22_DEFAULT_SEED})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=A22_DEFAULT_BATCH,
        help=f"number of independent worlds (default: {A22_DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=A22_DEFAULT_CANDIDATES,
        help=f"candidates per world (default: {A22_DEFAULT_CANDIDATES})",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=A22_DEFAULT_WARMUPS,
        help=(
            "number of complete warm-up episodes, excluded from statistics "
            f"(default: {A22_DEFAULT_WARMUPS})"
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=A22_DEFAULT_REPETITIONS,
        help=(
            "number of measured full episodes "
            f"(default: {A22_DEFAULT_REPETITIONS})"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/a22_cpu_timing.json"),
        help=(
            "JSON receipt destination "
            "(default: benchmarks/results/a22_cpu_timing.json)"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        receipt = run_benchmark(
            root=args.root,
            seed=args.seed,
            batch=args.batch,
            candidates=args.candidates,
            warmups=args.warmups,
            repetitions=args.repetitions,
        )
        write_receipt(receipt, args.output)
    except Exception as exc:
        parser.error(f"{type(exc).__name__}: {exc}")
        return 2

    print(f"A2.2 receipt: {args.output.resolve()}")
    print(f"Status: {receipt['status']}")

    summary = receipt.get("summary") or {}
    if receipt.get("valid") is True:
        print(
            "Median episode time: "
            f"{summary['median_episode_seconds']:.6f} s"
        )
        print(
            "Median world-steps/s: "
            f"{summary['median_world_steps_per_second']:.3f}"
        )
        print(
            "Measured repetitions: "
            f"{summary['valid_repetition_count']}/"
            f"{summary['measured_repetition_count']} valid"
        )
    else:
        print("No accepted throughput rates were produced.")
        if receipt.get("error"):
            print(f"Reason: {receipt['error']}")

    return 0 if receipt.get("valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
