"""Scoped CPU component profiling for the LAWLESS A2.2 environment.

Runs one fresh, full 2400-step A21 episode and profiles the exact callable
boundaries imported by lawless.environment.step:

    physics_step
    resolve_static_obstacles
    sense_lidar
    assemble_observations

The public step timer encloses these component timers. Component durations
are inclusive and nested: they must not be summed or treated as uninstrumented
baseline rates.

No production, A21, or timing-runner source is modified.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from benchmarks import environment_timing_a22 as timing


A21 = timing.a21

EPISODE_STEPS = int(timing.A21_EPISODE_STEPS)
DEFAULT_SEED = 7
DEFAULT_BATCH = 1
DEFAULT_CANDIDATES = 1
DEFAULT_OUTPUT = Path("benchmarks/results/a22_cpu_profile.json")

COMPONENT_NAMES = (
    "physics_step",
    "resolve_static_obstacles",
    "sense_lidar",
    "assemble_observations",
)

OWNER_MODULES = {
    "physics_step": "lawless.physics.simulator",
    "resolve_static_obstacles": "lawless.world.collision",
    "sense_lidar": "lawless.sensors.lidar",
    "assemble_observations": "lawless.sensors.observations",
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Strong references to module objects prevent object-ID reuse. A module
# whose restoration failed remains permanently contaminated in this process.
_ACTIVE_MODULES: set[Any] = set()
_CONTAMINATED_MODULES: set[Any] = set()


class ProfileError(RuntimeError):
    """The requested component profile failed one or more evidence gates."""


def _error_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _finite(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _clock_ns() -> int:
    value = time.perf_counter_ns()
    if not _finite(value):
        raise ProfileError("perf_counter_ns returned a non-finite value")
    return int(value)


def _elapsed_seconds(start_ns: Any, end_ns: Any, label: str) -> float:
    if not _finite(start_ns) or not _finite(end_ns):
        raise ProfileError(f"{label}: non-finite clock reading")

    elapsed_ns = float(end_ns) - float(start_ns)
    if not math.isfinite(elapsed_ns) or elapsed_ns < 0:
        raise ProfileError(f"{label}: invalid clock interval")

    seconds = elapsed_ns / 1_000_000_000.0
    if not math.isfinite(seconds) or seconds < 0:
        raise ProfileError(f"{label}: invalid elapsed seconds")
    return seconds


def _empty_timer() -> dict[str, Any]:
    return {
        "attempted_calls": 0,
        "successful_calls": 0,
        "inclusive_seconds": 0.0,
        "timing_valid": True,
        "error": None,
    }


def _empty_components() -> dict[str, dict[str, Any]]:
    return {name: _empty_timer() for name in COMPONENT_NAMES}


def _record_error(receipt: dict[str, Any], exc: BaseException) -> None:
    if receipt.get("error") is None:
        receipt["error"] = _error_text(exc)


def _accumulate_duration(
    receipt: dict[str, Any],
    start_ns: int,
    label: str,
) -> None:
    """Read the end clock and commit only a finite, nonnegative total."""
    try:
        end_ns = _clock_ns()
        elapsed = _elapsed_seconds(start_ns, end_ns, label)
        updated = float(receipt["inclusive_seconds"]) + elapsed
        if not math.isfinite(updated) or updated < 0:
            raise ProfileError(f"{label}: accumulated duration overflow")
        receipt["inclusive_seconds"] = updated
    except BaseException as exc:
        receipt["timing_valid"] = False
        _record_error(receipt, exc)
        raise


def _make_timed_wrapper(
    *,
    original: Callable[..., Any],
    receipt: dict[str, Any],
    label: str,
) -> Callable[..., Any]:
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        receipt["attempted_calls"] += 1

        try:
            start_ns = _clock_ns()
        except BaseException as exc:
            receipt["timing_valid"] = False
            _record_error(receipt, exc)
            raise

        original_error: BaseException | None = None
        result: Any = None

        try:
            result = original(*args, **kwargs)
            receipt["successful_calls"] += 1
        except BaseException as exc:
            original_error = exc
            _record_error(receipt, exc)

        try:
            _accumulate_duration(receipt, start_ns, label)
        except BaseException:
            # Do not replace an exception raised by the wrapped production
            # callable with a secondary clock failure.
            if original_error is None:
                raise

        if original_error is not None:
            raise original_error

        return result

    wrapped.__name__ = f"_a22_profiled_{label}"
    wrapped.__qualname__ = wrapped.__name__
    wrapped.__doc__ = f"Temporary inclusive A2.2 timer for {label}."
    wrapped.__a22_profile_wrapper__ = True
    return wrapped


def _load_step_module() -> Any:
    """Import the module explicitly; the package may export step as a function."""
    module = importlib.import_module("lawless.environment.step")

    expected_path = (
        Path(__file__).resolve().parents[1]
        / "lawless"
        / "environment"
        / "step.py"
    )

    module_path_value = getattr(module, "__file__", None)
    if not module_path_value:
        raise ProfileError("production step module has no file identity")

    try:
        actual_path = Path(module_path_value).resolve()
        expected_path = expected_path.resolve()
    except (OSError, RuntimeError, TypeError) as exc:
        raise ProfileError("production step module path is unsupported") from exc

    if actual_path != expected_path:
        raise ProfileError(
            "imported step module is not the repository production step.py"
        )

    if not callable(getattr(module, "step", None)):
        raise ProfileError("production step module has no callable step")

    for name in COMPONENT_NAMES:
        if not callable(getattr(module, name, None)):
            raise ProfileError(f"production step module has no callable {name}")

    return module


def _original_callables(step_module: Any) -> dict[str, Any]:
    """Capture exact imported objects and verify them against owner modules."""
    if step_module in _ACTIVE_MODULES:
        raise ProfileError("a profiler is already active for this module")
    if step_module in _CONTAMINATED_MODULES:
        raise ProfileError(
            "this step module is contaminated by a prior failed restoration"
        )

    originals = {
        "module_step": step_module.step,
        "a21_step": A21.step,
    }
    originals.update({
        name: getattr(step_module, name)
        for name in COMPONENT_NAMES
    })

    if originals["module_step"] is not originals["a21_step"]:
        raise ProfileError(
            "A21.step is not the exact production module step callable"
        )

    for name in COMPONENT_NAMES:
        owner = importlib.import_module(OWNER_MODULES[name])
        owner_name = {
            "physics_step": "step",
            "resolve_static_obstacles": "resolve_static_obstacles",
            "sense_lidar": "sense_lidar",
            "assemble_observations": "assemble_observations",
        }[name]
        owner_callable = getattr(owner, owner_name, None)

        if not callable(owner_callable):
            raise ProfileError(
                f"owning production module lacks callable {owner_name}"
            )

        if originals[name] is not owner_callable:
            raise ProfileError(
                f"{name} is not the exact callable imported from "
                f"{OWNER_MODULES[name]}"
            )

    for name, function in originals.items():
        if not callable(function):
            raise ProfileError(f"captured {name} is not callable")
        if getattr(function, "__a22_profile_wrapper__", False):
            raise ProfileError(f"{name} is already a profiler wrapper")

    return originals


def _restore_all(
    step_module: Any,
    originals: dict[str, Any],
) -> dict[str, Any]:
    """Attempt every restoration independently, then verify exact identities."""
    errors: list[str] = []

    restorations = [
        (
            "physics_step",
            lambda: setattr(
                step_module, "physics_step", originals["physics_step"]
            ),
        ),
        (
            "resolve_static_obstacles",
            lambda: setattr(
                step_module,
                "resolve_static_obstacles",
                originals["resolve_static_obstacles"],
            ),
        ),
        (
            "sense_lidar",
            lambda: setattr(
                step_module, "sense_lidar", originals["sense_lidar"]
            ),
        ),
        (
            "assemble_observations",
            lambda: setattr(
                step_module,
                "assemble_observations",
                originals["assemble_observations"],
            ),
        ),
        (
            "module_step",
            lambda: setattr(step_module, "step", originals["module_step"]),
        ),
        (
            "a21_step",
            lambda: setattr(A21, "step", originals["a21_step"]),
        ),
    ]

    for label, restore in restorations:
        try:
            restore()
        except BaseException as exc:
            errors.append(f"{label} restoration: {_error_text(exc)}")

    identities = {
        name: getattr(step_module, name, None) is originals[name]
        for name in COMPONENT_NAMES
    }
    identities["module_step"] = (
        getattr(step_module, "step", None) is originals["module_step"]
    )
    identities["a21_step"] = A21.step is originals["a21_step"]

    for name, restored in identities.items():
        if not restored:
            errors.append(f"{name} callable identity was not restored")

    verified = not errors and all(identities.values())

    if verified:
        _ACTIVE_MODULES.discard(step_module)
    else:
        # Never permit a later invocation to run as if this module were clean.
        _CONTAMINATED_MODULES.add(step_module)
        _ACTIVE_MODULES.add(step_module)

    return {
        "status": "verified" if verified else "failed",
        "identities": identities,
        "errors": errors,
    }


def _install_all(
    step_module: Any,
    originals: dict[str, Any],
    components: dict[str, dict[str, Any]],
    public_step: dict[str, Any],
) -> dict[str, Any]:
    """Install wrappers. Caller must restore originals after any attempt."""
    if step_module in _ACTIVE_MODULES:
        raise ProfileError("refusing nested profiler installation")
    if step_module in _CONTAMINATED_MODULES:
        raise ProfileError("refusing installation on a contaminated module")

    wrappers: dict[str, Any] = {}
    for name in COMPONENT_NAMES:
        wrappers[name] = _make_timed_wrapper(
            original=originals[name],
            receipt=components[name],
            label=name,
        )

    wrappers["module_step"] = _make_timed_wrapper(
        original=originals["module_step"],
        receipt=public_step,
        label="public_step",
    )

    _ACTIVE_MODULES.add(step_module)

    # Any setattr can fail after earlier assignments have succeeded.
    # The caller sets install_attempted before entering this function and
    # unconditionally invokes _restore_all in finally.
    for name in COMPONENT_NAMES:
        setattr(step_module, name, wrappers[name])

    step_module.step = wrappers["module_step"]
    A21.step = wrappers["module_step"]

    return wrappers


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _normalize_artifact_inventory(
    raw: Any,
    *,
    required_paths: dict[str, str] | None = None,
) -> dict[str, str]:
    """Require verified file records with repository-relative paths/digests."""
    if not isinstance(raw, dict) or not raw:
        raise ProfileError("base artifact inventory is missing or empty")

    normalized: dict[str, str] = {}

    for path, record in raw.items():
        if not isinstance(path, str) or not path:
            raise ProfileError("artifact inventory contains an invalid path")

        if not isinstance(record, dict):
            raise ProfileError(
                f"artifact {path} lacks a structured hash record"
            )

        if record.get("status") != "verified":
            raise ProfileError(
                f"artifact {path} does not have verified status"
            )

        recorded_path = record.get("path")
        if not isinstance(recorded_path, str) or not recorded_path:
            raise ProfileError(f"artifact {path} has no recorded path")

        recorded_path_obj = Path(recorded_path)
        if (
            recorded_path_obj.is_absolute()
            or ".." in recorded_path_obj.parts
            or recorded_path_obj.as_posix() != recorded_path
        ):
            raise ProfileError(
                f"artifact {path} has an invalid repository-relative path"
            )

        if required_paths is not None:
            expected_path = required_paths.get(path)
            if expected_path is not None and recorded_path != expected_path:
                raise ProfileError(
                    f"artifact {path} path does not match its required file"
                )

        size_bytes = record.get("size_bytes")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ProfileError(
                f"artifact {path} has invalid recorded file size"
            )

        digest = record.get("sha256")
        if not _valid_sha256(digest):
            raise ProfileError(
                f"artifact {path} does not have a valid SHA-256 digest"
            )

        normalized[path] = digest

    if required_paths is not None:
        missing = sorted(set(required_paths) - set(normalized))
        if missing:
            raise ProfileError(
                "required base artifact records are missing: "
                + ", ".join(missing)
            )

    return dict(sorted(normalized.items()))


def _artifact_hashes(root: Path) -> dict[str, str]:
    """Validate base artifacts and add the profiler and focused-test files."""
    root = root.resolve()
    raw = timing._artifact_hashes(root)

    required_base = {
        "a21_benchmark": "benchmarks/environment_benchmark.py",
        "a22_runner": "benchmarks/environment_timing_a22.py",
        "a22_tests": "tests/test_environment_benchmark_a22.py",
    }
    normalized = _normalize_artifact_inventory(
        raw,
        required_paths=required_base,
    )

    required_paths = (
        Path("benchmarks/environment_profile_a22.py"),
        Path("tests/test_environment_profile_a22.py"),
    )

    for relative in required_paths:
        relative_text = relative.as_posix()
        path = root / relative

        if not path.is_file():
            raise ProfileError(
                f"required evidence artifact is missing: {relative_text}"
            )

        record = timing._sha256_file(path, root=root)
        extra = _normalize_artifact_inventory(
            {relative_text: record},
            required_paths={relative_text: relative_text},
        )
        digest = extra[relative_text]
        if not _valid_sha256(digest):
            raise ProfileError(
                f"required evidence artifact has invalid digest: {relative_text}"
            )

        normalized[relative_text] = digest

    return dict(sorted(normalized.items()))


def _source_identity(root: Path) -> dict[str, Any]:
    identity = timing._get_source_identity(root)

    if not isinstance(identity, dict):
        raise ProfileError("source identity helper returned a non-dictionary")
    if identity.get("identity_complete") is not True:
        raise ProfileError("source identity is incomplete")

    preservation = identity.get("frozen_p0_preservation")
    if not isinstance(preservation, dict):
        raise ProfileError("frozen P0 preservation evidence is missing")
    if preservation.get("status") != "verified":
        raise ProfileError("frozen P0 preservation is not verified")
    if preservation.get("file_count") != 20:
        raise ProfileError("frozen P0 inventory does not contain 20 files")
    if preservation.get("compared_file_count") != 20:
        raise ProfileError("not all 20 frozen P0 files were compared")
    if preservation.get("mismatched_files") != []:
        raise ProfileError("one or more frozen P0 files have changed")

    return identity


def _identity_fingerprint(identity: dict[str, Any]) -> Any:
    return timing._identity_fingerprint(identity)


def _merge_partial_progress(
    receipt: dict[str, Any],
    environment: Any,
    transitions_attempted: int,
    public_step_attempted: int,
    public_step_returned: int,
) -> None:
    """Merge runner world-state evidence without overwriting known call counts."""
    progress = receipt["progress"]
    progress["transitions_attempted"] = transitions_attempted
    progress["public_step_calls_attempted"] = public_step_attempted
    progress["public_step_calls_returned"] = public_step_returned

    if environment is None:
        return

    try:
        partial = timing._partial_progress(environment)
        if not isinstance(partial, dict):
            progress["partial_progress_error"] = (
                "timing._partial_progress returned a non-dictionary"
            )
            return

        protected = {
            "transitions_attempted",
            "public_step_calls_attempted",
            "public_step_calls_returned",
        }

        for key, value in partial.items():
            if key not in protected:
                progress[key] = value
    except BaseException as exc:
        progress["partial_progress_error"] = _error_text(exc)


def _validate_timer_receipt(label: str, timer: dict[str, Any]) -> None:
    if timer["attempted_calls"] != timer["successful_calls"]:
        raise ProfileError(
            f"{label}: attempted and successful call counts differ"
        )
    if not timer["timing_valid"]:
        raise ProfileError(f"{label}: timing was invalid")
    if timer["error"] is not None:
        raise ProfileError(f"{label}: {timer['error']}")
    if not _finite(timer["inclusive_seconds"]):
        raise ProfileError(f"{label}: accumulated duration is non-finite")
    if float(timer["inclusive_seconds"]) < 0:
        raise ProfileError(f"{label}: accumulated duration is negative")


def _validate_profile_timers(
    components: dict[str, dict[str, Any]],
    public_step: dict[str, Any],
) -> None:
    expected = EPISODE_STEPS

    _validate_timer_receipt("public_step", public_step)
    if public_step["attempted_calls"] != expected:
        raise ProfileError("public step attempted-call count is not 2400")
    if public_step["successful_calls"] != expected:
        raise ProfileError("public step successful-call count is not 2400")

    for name in COMPONENT_NAMES:
        timer = components[name]
        _validate_timer_receipt(name, timer)

        if timer["attempted_calls"] != expected:
            raise ProfileError(
                f"{name} attempted-call count is not 2400"
            )
        if timer["successful_calls"] != expected:
            raise ProfileError(
                f"{name} successful-call count is not 2400"
            )


def _base_receipt(
    *,
    seed: int,
    batch: int,
    candidates: int,
    root: Path,
) -> dict[str, Any]:
    return {
        "schema": "lawless.a22.component_profile.v2",
        "status": "invalid",
        "accepted_profile": False,
        "workload": {
            "seed": seed,
            "batch": batch,
            "candidates": candidates,
            "device": "cpu",
            "dtype": "torch.float32",
            "episode_steps": EPISODE_STEPS,
            "required_public_step_calls": EPISODE_STEPS,
            "required_world_transitions": EPISODE_STEPS,
            "warmup_repetitions": 0,
            "profiled_repetitions": 1,
        },
        "timing_method": {
            "clock": "time.perf_counter_ns",
            "component_intervals": "inclusive wall-clock wrapper intervals",
            "public_step_interval": (
                "inclusive complete public-step wrapper interval"
            ),
            "nesting": {
                "public_step_contains_components": True,
                "component_intervals_are_inclusive": True,
                "component_totals_must_not_be_summed": True,
            },
            "overhead": {
                "method": (
                    "Wrapper dispatch and clock-read overhead remain included "
                    "in each recorded interval."
                ),
                "subtracted": False,
                "separately_quantified": False,
            },
            "profile_is_uninstrumented_baseline": False,
            "baseline_rate_claim": False,
        },
        "reset_seconds": None,
        "controller_and_public_step_loop_seconds": None,
        "public_step": _empty_timer(),
        "components": _empty_components(),
        "progress": {
            "transitions_attempted": 0,
            "public_step_calls_attempted": 0,
            "public_step_calls_returned": 0,
            "world_step_counts": None,
            "minimum_world_step_count": None,
            "world_completed": None,
            "all_completed": None,
        },
        "validation": {
            "arguments": "not_run",
            "source_before": "not_run",
            "artifacts_before": "not_run",
            "config": "not_run",
            "initial_environment": "not_run",
            "reset_observations": "not_run",
            "final_observations": "not_run",
            "final_environment": "not_run",
            "component_attribution": "not_run",
            "restoration": "not_run",
            "source_after": "not_run",
            "artifacts_after": "not_run",
        },
        "source_before": None,
        "source_after": None,
        "source_unchanged": False,
        "artifacts_before": None,
        "artifacts_after": None,
        "artifacts_unchanged": False,
        "restoration": {
            "status": "not_attempted",
            "identities": {},
            "errors": [],
        },
        "runtime": None,
        "final_validation": None,
        "error": None,
        "root": str(root),
    }


def _sanitize_json(value: Any, path: str = "$") -> tuple[Any, list[str]]:
    """Replace non-finite raw numbers with null and report every occurrence."""
    errors: list[str] = []

    if isinstance(value, float):
        if not math.isfinite(value):
            errors.append(f"{path}: non-finite float replaced with null")
            return None, errors
        return value, errors

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            safe_item, item_errors = _sanitize_json(item, f"{path}.{key}")
            result[str(key)] = safe_item
            errors.extend(item_errors)
        return result, errors

    if isinstance(value, (list, tuple)):
        result_list = []
        for index, item in enumerate(value):
            safe_item, item_errors = _sanitize_json(item, f"{path}[{index}]")
            result_list.append(safe_item)
            errors.extend(item_errors)
        return result_list, errors

    if isinstance(value, (str, int, bool)) or value is None:
        return value, errors

    # Tensor, Path, and other project values should already have been passed
    # through the runner's json_safe helper. Do not silently stringify them.
    errors.append(f"{path}: unsupported JSON value {type(value).__name__}")
    return None, errors


def _strict_json_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Sanitize raw evidence and prove strict JSON serialization succeeds."""
    sanitized, errors = _sanitize_json(receipt)

    if not isinstance(sanitized, dict):
        raise ProfileError("sanitized receipt is not a dictionary")

    if errors:
        sanitized["status"] = "invalid"
        sanitized["accepted_profile"] = False
        existing = sanitized.get("error")
        detail = "; ".join(errors)
        sanitized["error"] = (
            f"{existing}; {detail}" if existing else detail
        )

    encoded = json.dumps(
        sanitized,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    # Verify the emitted representation is parseable and remains a mapping.
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise ProfileError("strict JSON round-trip did not return a mapping")

    return decoded


def profile_episode(
    *,
    seed: int = DEFAULT_SEED,
    batch: int = DEFAULT_BATCH,
    candidates: int = DEFAULT_CANDIDATES,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Profile one fresh full-length A21 episode on CPU/float32."""
    root = Path(__file__).resolve().parents[1]

    receipt = _base_receipt(
        seed=seed,
        batch=batch,
        candidates=candidates,
        root=root,
    )

    if output_path is None:
        output = root / DEFAULT_OUTPUT
    else:
        output = Path(output_path)
        if not output.is_absolute():
            output = root / output

    step_module = None
    originals = None
    wrappers = None
    install_attempted = False

    environment = None
    final_observations = None
    initial_geometry = None
    initial_gravity = None
    targets = None

    reset_start_ns = None
    reset_end_ns = None
    loop_start_ns = None
    loop_end_ns = None

    transitions_attempted = 0
    public_step_attempted = 0
    public_step_returned = 0

    try:
        # Use the timing runner's own argument contracts. In particular,
        # seed validation preserves its supported zero-seed semantics.
        timing._validate_arguments(
            batch=batch,
            candidates=candidates,
            seed=seed,
            warmups=1,
            repetitions=1,
        )

        if batch != 1 or candidates != 1:
            raise ProfileError("component profiling currently requires B1/N1")

        receipt["validation"]["arguments"] = "passed"
        receipt["runtime"] = timing._runtime_metadata()

        receipt["source_before"] = _source_identity(root)
        receipt["validation"]["source_before"] = "verified"

        receipt["artifacts_before"] = _artifact_hashes(root)
        receipt["validation"]["artifacts_before"] = "verified"

        step_module = _load_step_module()
        originals = _original_callables(step_module)

        config = A21.create_a21_config(seed)
        timing._validate_config(config)
        receipt["validation"]["config"] = "passed"

        reset_start_ns = _clock_ns()
        environment, reset_observations, reset_info = A21.reset(
            config,
            batch,
            candidates,
            device="cpu",
            dtype=torch.float32,
        )
        reset_end_ns = _clock_ns()

        receipt["reset_seconds"] = _elapsed_seconds(
            reset_start_ns,
            reset_end_ns,
            "reset",
        )

        timing._validate_initial_environment(
            environment,
            batch=batch,
            candidates=candidates,
            config=config,
        )
        receipt["validation"]["initial_environment"] = "passed"

        timing._validate_observations(
            reset_observations,
            batch=batch,
            candidates=candidates,
            label="reset observations",
        )
        receipt["validation"]["reset_observations"] = "passed"

        initial_geometry = A21.geometry_inventory(environment)
        initial_gravity = A21.a21_gravity_snapshot(environment)
        targets = timing._capture_target_positions(environment)

        # Set this before the first setattr. If any later setattr raises,
        # finally restores every original independently.
        install_attempted = True
        wrappers = _install_all(
            step_module,
            originals,
            receipt["components"],
            receipt["public_step"],
        )

        loop_start_ns = _clock_ns()

        for _ in range(EPISODE_STEPS):
            transitions_attempted += 1
            actions = A21.a21_controller_actions(environment, targets)

            # Count the call before entering it, so a throwing public step
            # remains visible as attempted but not returned.
            public_step_attempted += 1
            (
                environment,
                final_observations,
                rewards,
                terminated,
                truncated,
                info,
            ) = A21.step(environment, actions)

            public_step_returned += 1

        loop_end_ns = _clock_ns()
        receipt["controller_and_public_step_loop_seconds"] = (
            _elapsed_seconds(
                loop_start_ns,
                loop_end_ns,
                "controller/public-step loop",
            )
        )

        _merge_partial_progress(
            receipt,
            environment,
            transitions_attempted,
            public_step_attempted,
            public_step_returned,
        )

        timing._validate_observations(
            final_observations,
            batch=batch,
            candidates=candidates,
            label="final observations",
        )
        receipt["validation"]["final_observations"] = "passed"

        receipt["final_validation"] = timing._validate_final_environment(
            environment,
            batch=batch,
            candidates=candidates,
            config=config,
            initial_gravity=initial_gravity,
            initial_geometry=initial_geometry,
            final_observations=final_observations,
        )
        receipt["validation"]["final_environment"] = "passed"

        _validate_profile_timers(
            receipt["components"],
            receipt["public_step"],
        )
        receipt["validation"]["component_attribution"] = "passed"

    except BaseException as exc:
        _record_error(receipt, exc)

    finally:
        _merge_partial_progress(
            receipt,
            environment,
            transitions_attempted,
            public_step_attempted,
            public_step_returned,
        )

        if install_attempted and step_module is not None and originals is not None:
            try:
                receipt["restoration"] = _restore_all(
                    step_module,
                    originals,
                )
            except BaseException as exc:
                # A failure of the restoration verifier is itself a failed
                # restoration. Retain the module in the contamination guard.
                _CONTAMINATED_MODULES.add(step_module)
                _ACTIVE_MODULES.add(step_module)
                receipt["restoration"] = {
                    "status": "failed",
                    "identities": {},
                    "errors": [_error_text(exc)],
                }

            receipt["validation"]["restoration"] = (
                receipt["restoration"]["status"]
            )

        elif step_module is not None and originals is not None:
            # No installation was attempted. Still check that reset/setup
            # did not alter any inspected production callable identities.
            identities = {
                name: getattr(step_module, name, None) is originals[name]
                for name in COMPONENT_NAMES
            }
            identities["module_step"] = (
                getattr(step_module, "step", None)
                is originals["module_step"]
            )
            identities["a21_step"] = A21.step is originals["a21_step"]

            restored = all(identities.values())
            receipt["restoration"] = {
                "status": "verified" if restored else "failed",
                "identities": identities,
                "errors": [] if restored else [
                    "callable identity changed before installation"
                ],
            }
            receipt["validation"]["restoration"] = (
                "verified" if restored else "failed"
            )

        else:
            receipt["restoration"] = {
                "status": "not_attempted",
                "identities": {},
                "errors": [],
            }
            receipt["validation"]["restoration"] = "not_attempted"

    # Source and artifacts are checked only after attempted hooks have been
    # restored. These operations are outside all profile intervals.
    try:
        receipt["source_after"] = _source_identity(root)
        receipt["source_unchanged"] = (
            _identity_fingerprint(receipt["source_before"])
            == _identity_fingerprint(receipt["source_after"])
        )
        if not receipt["source_unchanged"]:
            raise ProfileError("source identity changed during profiling")
        receipt["validation"]["source_after"] = "verified"
    except BaseException as exc:
        _record_error(receipt, exc)
        receipt["source_unchanged"] = False
        receipt["validation"]["source_after"] = "failed"

    try:
        receipt["artifacts_after"] = _artifact_hashes(root)
        receipt["artifacts_unchanged"] = (
            receipt["artifacts_before"] == receipt["artifacts_after"]
        )
        if not receipt["artifacts_unchanged"]:
            raise ProfileError("artifact hashes changed during profiling")
        receipt["validation"]["artifacts_after"] = "verified"
    except BaseException as exc:
        _record_error(receipt, exc)
        receipt["artifacts_unchanged"] = False
        receipt["validation"]["artifacts_after"] = "failed"

    # Final acceptance is conjunctive. No profile is accepted if any
    # validation, timing, source, artifact, or restoration gate failed.
    try:
        if receipt["error"] is not None:
            raise ProfileError(receipt["error"])

        if transitions_attempted != EPISODE_STEPS:
            raise ProfileError(
                f"only {transitions_attempted} transitions were attempted"
            )
        if public_step_attempted != EPISODE_STEPS:
            raise ProfileError(
                f"only {public_step_attempted} public calls were attempted"
            )
        if public_step_returned != EPISODE_STEPS:
            raise ProfileError(
                f"only {public_step_returned} public calls returned"
            )

        _validate_profile_timers(
            receipt["components"],
            receipt["public_step"],
        )

        if receipt["restoration"].get("status") != "verified":
            raise ProfileError("callable restoration was not verified")

        if receipt["source_unchanged"] is not True:
            raise ProfileError("source identity did not remain unchanged")

        if receipt["artifacts_unchanged"] is not True:
            raise ProfileError("artifact hashes did not remain unchanged")

        if receipt["validation"]["final_environment"] != "passed":
            raise ProfileError("final environment validation did not pass")

        if receipt["reset_seconds"] is None:
            raise ProfileError("reset duration is unavailable")
        if not _finite(receipt["reset_seconds"]):
            raise ProfileError("reset duration is non-finite")
        if float(receipt["reset_seconds"]) < 0:
            raise ProfileError("reset duration is negative")

        loop_seconds = receipt["controller_and_public_step_loop_seconds"]
        if loop_seconds is None or not _finite(loop_seconds):
            raise ProfileError("profile loop duration is unavailable or invalid")
        if float(loop_seconds) < 0:
            raise ProfileError("profile loop duration is negative")

        receipt["status"] = "valid"
        receipt["accepted_profile"] = True

    except BaseException as exc:
        receipt["status"] = "invalid"
        receipt["accepted_profile"] = False
        _record_error(receipt, exc)

    # Sanitize raw non-finite values and require a strict JSON round-trip.
    try:
        receipt = _strict_json_receipt(receipt)
        if receipt.get("error") is not None:
            receipt["status"] = "invalid"
            receipt["accepted_profile"] = False
    except BaseException as exc:
        receipt["status"] = "invalid"
        receipt["accepted_profile"] = False
        _record_error(receipt, exc)

    # Verify strict serialization once more after any final status/error edits.
    try:
        json.dumps(receipt, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        # Keep a minimal strict-JSON failure receipt rather than emitting
        # invalid JSON or losing the reason the profile was rejected.
        receipt = {
            "schema": "lawless.a22.component_profile.v2",
            "status": "invalid",
            "accepted_profile": False,
            "error": (
                receipt.get("error")
                or f"strict JSON serialization failed: {_error_text(exc)}"
            ),
            "restoration": receipt.get("restoration"),
            "progress": receipt.get("progress"),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    timing.write_receipt(receipt, output)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one complete A21 CPU episode at the exact production "
            "physics, collision, lidar, observation, and public-step "
            "callable boundaries."
        )
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument(
        "--candidates",
        type=int,
        default=DEFAULT_CANDIDATES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        receipt = profile_episode(
            seed=args.seed,
            batch=args.batch,
            candidates=args.candidates,
            output_path=args.output,
        )
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "status": "invalid",
                    "accepted_profile": False,
                    "error": _error_text(exc),
                },
                sort_keys=True,
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2

    print(
        json.dumps(
            {
                "status": receipt.get("status"),
                "accepted_profile": receipt.get("accepted_profile"),
                "output": str(args.output),
                "error": receipt.get("error"),
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if receipt.get("accepted_profile") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
