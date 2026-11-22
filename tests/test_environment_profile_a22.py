"""Focused tests for the LAWLESS A2.2 CPU component profiler."""

from __future__ import annotations

import functools
import importlib
import json
import math
import types
from pathlib import Path
from typing import Any, Callable

import pytest

from benchmarks import environment_profile_a22 as profiler


BASE_ARTIFACT_PATHS = {
    "a21_benchmark": "benchmarks/environment_benchmark.py",
    "a22_runner": "benchmarks/environment_timing_a22.py",
    "a22_tests": "tests/test_environment_benchmark_a22.py",
}

OBSERVATIONS = object()


def _hash_record(path: str, digest: str | None = None) -> dict[str, Any]:
    return {
        "path": path,
        "status": "verified",
        "sha256": digest or ("a" * 64),
        "size_bytes": 123,
    }


def _strict_load(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise AssertionError(f"non-standard JSON constant: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


class SyntheticStepModule(types.ModuleType):
    """Hashable isolated step module with controllable setattr failures."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._fail_install_name: str | None = None
        self._install_failure_used = False
        self._fail_restore_name: str | None = None
        self._restore_original: Any = None

    def __setattr__(self, name: str, value: Any) -> None:
        fail_install_name = getattr(self, "_fail_install_name", None)

        if (
            name == fail_install_name
            and not getattr(self, "_install_failure_used", False)
        ):
            object.__setattr__(self, "_install_failure_used", True)
            raise RuntimeError(f"synthetic install failure at {name}")

        if (
            name == getattr(self, "_fail_restore_name", None)
            and value is getattr(self, "_restore_original", None)
            and getattr(self, "_restore_original", None) is not None
        ):
            raise RuntimeError(f"synthetic restoration failure at {name}")

        super().__setattr__(name, value)


@pytest.fixture
def synthetic_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Build a complete, identity-consistent synthetic profiling environment."""

    def build(
        *,
        failure: str | None = None,
        install_failure: bool = False,
        restore_failure: bool = False,
        source_change: bool = False,
        artifact_change: bool = False,
    ) -> dict[str, Any]:
        module = SyntheticStepModule("lawless.environment.step.synthetic")
        owner_modules: dict[str, types.ModuleType] = {}

        component_calls = {name: 0 for name in profiler.COMPONENT_NAMES}

        def make_component(name: str) -> Callable[..., Any]:
            def component(*args: Any, **kwargs: Any) -> Any:
                component_calls[name] += 1
                if failure == name:
                    raise RuntimeError(f"synthetic {name} failure")
                return None

            component.__name__ = name
            return component

        components = {
            name: make_component(name)
            for name in profiler.COMPONENT_NAMES
        }

        environment = types.SimpleNamespace(steps=0)

        def public_step(env: Any, actions: Any):
            for name in profiler.COMPONENT_NAMES:
                getattr(module, name)(env, actions)

            if failure == "public_step":
                raise RuntimeError("synthetic public step failure")

            env.steps += 1
            return env, OBSERVATIONS, None, None, None, {}

        public_step.__name__ = "step"

        for name in profiler.COMPONENT_NAMES:
            setattr(module, name, components[name])

            owner = types.ModuleType(profiler.OWNER_MODULES[name])
            owner_name = (
                "step" if name == "physics_step" else name
            )
            setattr(owner, owner_name, components[name])
            owner_modules[profiler.OWNER_MODULES[name]] = owner

        module.step = public_step

        if install_failure:
            module._fail_install_name = "sense_lidar"

        if restore_failure:
            module._fail_restore_name = "sense_lidar"
            module._restore_original = components["sense_lidar"]

        # _original_callables must see the exact same function objects in
        # the step module and their corresponding owning production modules.
        real_import_module = importlib.import_module

        def synthetic_import_module(
            name: str,
            package: str | None = None,
        ) -> Any:
            if name in owner_modules:
                return owner_modules[name]
            return real_import_module(name, package)

        monkeypatch.setattr(
            profiler.importlib,
            "import_module",
            synthetic_import_module,
        )
        monkeypatch.setattr(profiler, "_load_step_module", lambda: module)

        A21 = profiler.A21
        monkeypatch.setattr(A21, "step", public_step)
        monkeypatch.setattr(
            A21,
            "create_a21_config",
            lambda seed: {"seed": seed},
        )

        def reset(
            config: Any,
            batch: int,
            candidates: int,
            *,
            device: str,
            dtype: Any,
        ):
            if failure == "reset":
                raise RuntimeError("synthetic reset failure")
            return environment, OBSERVATIONS, {}

        monkeypatch.setattr(A21, "reset", reset)
        monkeypatch.setattr(
            A21,
            "geometry_inventory",
            lambda env: {"geometry": "unchanged"},
        )
        monkeypatch.setattr(
            A21,
            "a21_gravity_snapshot",
            lambda env: {"gravity": "initial"},
        )
        monkeypatch.setattr(
            A21,
            "a21_controller_actions",
            lambda env, targets: object(),
        )

        monkeypatch.setattr(
            profiler.timing,
            "_runtime_metadata",
            lambda: {"runtime": "synthetic-test"},
        )
        monkeypatch.setattr(
            profiler.timing,
            "_validate_config",
            lambda config: None,
        )
        monkeypatch.setattr(
            profiler.timing,
            "_validate_initial_environment",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            profiler.timing,
            "_validate_observations",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            profiler.timing,
            "_validate_final_environment",
            lambda *args, **kwargs: {"status": "verified"},
        )
        monkeypatch.setattr(
            profiler.timing,
            "_capture_target_positions",
            lambda env: object(),
        )
        monkeypatch.setattr(
            profiler.timing,
            "_partial_progress",
            lambda env: {
                # These deliberately unknown values must not overwrite the
                # profiler's exact call counters.
                "transitions_attempted": -100,
                "public_step_calls_attempted": -200,
                "public_step_calls_returned": -300,
                "world_step_counts": [env.steps],
                "world_completed": True,
            },
        )

        source_counter = {"calls": 0}

        def source_identity(root: Path) -> dict[str, Any]:
            source_counter["calls"] += 1
            fingerprint = "source-before"
            if source_change and source_counter["calls"] >= 2:
                fingerprint = "source-after"
            return {
                "identity_complete": True,
                "fingerprint": fingerprint,
                "frozen_p0_preservation": {
                    "status": "verified",
                    "file_count": 20,
                    "compared_file_count": 20,
                    "mismatched_files": [],
                },
            }

        monkeypatch.setattr(profiler, "_source_identity", source_identity)
        monkeypatch.setattr(
            profiler,
            "_identity_fingerprint",
            lambda identity: identity["fingerprint"],
        )

        artifact_counter = {"calls": 0}

        def artifact_hashes(root: Path) -> dict[str, str]:
            artifact_counter["calls"] += 1
            digest = "a" * 64
            if artifact_change and artifact_counter["calls"] >= 2:
                digest = "b" * 64
            return {"synthetic_artifact": digest}

        monkeypatch.setattr(profiler, "_artifact_hashes", artifact_hashes)

        # Each fixture uses an isolated module object. Clean up both global
        # guards even if a test intentionally exercises failed restoration.
        profiler._ACTIVE_MODULES.discard(module)
        profiler._CONTAMINATED_MODULES.discard(module)

        return {
            "module": module,
            "owners": owner_modules,
            "components": components,
            "component_calls": component_calls,
            "public_step": public_step,
            "environment": environment,
            "output": tmp_path / "synthetic-profile.json",
        }

    return build


def _assert_synthetic_identities_restored(ctx: dict[str, Any]) -> None:
    module = ctx["module"]
    assert module.step is ctx["public_step"]
    assert profiler.A21.step is ctx["public_step"]

    for name, original in ctx["components"].items():
        assert getattr(module, name) is original

    assert module not in profiler._ACTIVE_MODULES
    assert module not in profiler._CONTAMINATED_MODULES


def test_synthetic_profile_has_positive_attribution_counts_nesting_and_json(
    synthetic_factory,
):
    ctx = synthetic_factory()

    receipt = profiler.profile_episode(
        seed=0,
        batch=1,
        candidates=1,
        output_path=ctx["output"],
    )

    assert receipt["status"] == "valid"
    assert receipt["accepted_profile"] is True
    assert receipt["workload"]["seed"] == 0
    assert receipt["workload"]["episode_steps"] == 2400
    assert receipt["workload"]["warmup_repetitions"] == 0
    assert receipt["workload"]["profiled_repetitions"] == 1

    assert receipt["validation"]["arguments"] == "passed"
    assert receipt["validation"]["component_attribution"] == "passed"
    assert receipt["validation"]["restoration"] == "verified"
    assert receipt["validation"]["source_after"] == "verified"
    assert receipt["validation"]["artifacts_after"] == "verified"

    assert receipt["progress"]["transitions_attempted"] == 2400
    assert receipt["progress"]["public_step_calls_attempted"] == 2400
    assert receipt["progress"]["public_step_calls_returned"] == 2400

    # The timing helper's deliberately false counts did not overwrite the
    # profiler's exact counters.
    assert receipt["progress"]["transitions_attempted"] != -100
    assert receipt["progress"]["public_step_calls_attempted"] != -200
    assert receipt["progress"]["public_step_calls_returned"] != -300

    public = receipt["public_step"]
    assert public["attempted_calls"] == 2400
    assert public["successful_calls"] == 2400
    assert public["timing_valid"] is True
    assert public["inclusive_seconds"] > 0

    for name in profiler.COMPONENT_NAMES:
        timer = receipt["components"][name]
        assert timer["attempted_calls"] == 2400
        assert timer["successful_calls"] == 2400
        assert timer["timing_valid"] is True
        assert timer["error"] is None
        assert timer["inclusive_seconds"] > 0

        # Every component interval is nested within the corresponding
        # public-step interval. Component totals are inclusive, not additive.
        assert public["inclusive_seconds"] >= timer["inclusive_seconds"]

    assert receipt["timing_method"]["nesting"] == {
        "public_step_contains_components": True,
        "component_intervals_are_inclusive": True,
        "component_totals_must_not_be_summed": True,
    }
    assert receipt["timing_method"]["profile_is_uninstrumented_baseline"] is False
    assert receipt["timing_method"]["baseline_rate_claim"] is False
    assert receipt["timing_method"]["overhead"]["subtracted"] is False

    assert receipt["reset_seconds"] >= 0
    assert receipt["controller_and_public_step_loop_seconds"] >= 0

    _assert_synthetic_identities_restored(ctx)

    assert ctx["output"].is_file()
    persisted = _strict_load(ctx["output"])
    assert persisted == receipt
    json.dumps(persisted, allow_nan=False)


@pytest.mark.parametrize(
    ("failure", "expected_attempted", "expected_returned"),
    [
        ("reset", 0, 0),
        ("assemble_observations", 1, 0),
        ("public_step", 1, 0),
    ],
)
def test_profile_failure_paths_keep_attempted_and_returned_counts_and_restore(
    synthetic_factory,
    failure,
    expected_attempted,
    expected_returned,
):
    ctx = synthetic_factory(failure=failure)

    receipt = profiler.profile_episode(
        seed=7,
        batch=1,
        candidates=1,
        output_path=ctx["output"],
    )

    assert receipt["status"] == "invalid"
    assert receipt["accepted_profile"] is False
    assert receipt["error"] is not None
    assert receipt["restoration"]["status"] == "verified"
    assert receipt["validation"]["restoration"] == "verified"

    assert receipt["progress"]["public_step_calls_attempted"] == expected_attempted
    assert receipt["progress"]["public_step_calls_returned"] == expected_returned

    if failure == "reset":
        assert receipt["progress"]["transitions_attempted"] == 0
        assert receipt["reset_seconds"] is None
        assert receipt["public_step"]["attempted_calls"] == 0
        assert receipt["validation"]["initial_environment"] == "not_run"
    elif failure == "assemble_observations":
        assert receipt["progress"]["transitions_attempted"] == 1
        assert receipt["public_step"]["attempted_calls"] == 1
        assert receipt["public_step"]["successful_calls"] == 0
        assert receipt["components"]["assemble_observations"]["attempted_calls"] == 1
        assert receipt["components"]["assemble_observations"]["successful_calls"] == 0
        assert receipt["components"]["physics_step"]["successful_calls"] == 1
        assert receipt["components"]["resolve_static_obstacles"]["successful_calls"] == 1
        assert receipt["components"]["sense_lidar"]["successful_calls"] == 1
    else:
        assert receipt["progress"]["transitions_attempted"] == 1
        assert receipt["public_step"]["attempted_calls"] == 1
        assert receipt["public_step"]["successful_calls"] == 0
        for name in profiler.COMPONENT_NAMES:
            assert receipt["components"][name]["attempted_calls"] == 1
            assert receipt["components"][name]["successful_calls"] == 1

    _assert_synthetic_identities_restored(ctx)
    assert _strict_load(ctx["output"]) == receipt


def test_partial_installation_failure_restores_every_original_and_clears_guard(
    synthetic_factory,
):
    ctx = synthetic_factory(install_failure=True)
    module = ctx["module"]

    receipt = profiler.profile_episode(
        seed=7,
        batch=1,
        candidates=1,
        output_path=ctx["output"],
    )

    assert receipt["status"] == "invalid"
    assert receipt["accepted_profile"] is False
    assert "synthetic install failure at sense_lidar" in receipt["error"]

    # The install failed after earlier component assignments. The finally
    # path must still attempt all six restorations.
    assert receipt["restoration"]["status"] == "verified"
    assert all(receipt["restoration"]["identities"].values())
    _assert_synthetic_identities_restored(ctx)


def test_restoration_failure_attempts_all_objects_and_marks_module_contaminated(
    synthetic_factory,
):
    ctx = synthetic_factory(restore_failure=True)
    module = ctx["module"]

    receipt = profiler.profile_episode(
        seed=7,
        batch=1,
        candidates=1,
        output_path=ctx["output"],
    )

    assert receipt["status"] == "invalid"
    assert receipt["accepted_profile"] is False
    assert receipt["restoration"]["status"] == "failed"

    identities = receipt["restoration"]["identities"]
    assert identities["physics_step"] is True
    assert identities["resolve_static_obstacles"] is True
    assert identities["sense_lidar"] is False
    assert identities["assemble_observations"] is True
    assert identities["module_step"] is True
    assert identities["a21_step"] is True

    # Restoration continued after the sense_lidar setter raised.
    assert module.physics_step is ctx["components"]["physics_step"]
    assert module.resolve_static_obstacles is ctx["components"]["resolve_static_obstacles"]
    assert module.assemble_observations is ctx["components"]["assemble_observations"]
    assert module.step is ctx["public_step"]
    assert profiler.A21.step is ctx["public_step"]

    assert module in profiler._ACTIVE_MODULES
    assert module in profiler._CONTAMINATED_MODULES

    # _original_callables checks active first. Remove only the active marker
    # here to prove the independent contamination guard also rejects it.
    profiler._ACTIVE_MODULES.discard(module)
    with pytest.raises(profiler.ProfileError, match="contaminated"):
        profiler._original_callables(module)

    # This synthetic module is never the real production module. Remove its
    # process-local guard entries so no later test inherits this fixture.
    profiler._ACTIVE_MODULES.discard(module)
    profiler._CONTAMINATED_MODULES.discard(module)


def test_original_callables_requires_exact_owner_module_identity(
    synthetic_factory,
):
    ctx = synthetic_factory()
    module = ctx["module"]

    originals = profiler._original_callables(module)
    assert originals["module_step"] is ctx["public_step"]
    assert originals["a21_step"] is ctx["public_step"]

    for name in profiler.COMPONENT_NAMES:
        owner = ctx["owners"][profiler.OWNER_MODULES[name]]
        owner_name = "step" if name == "physics_step" else name
        assert originals[name] is getattr(owner, owner_name)
        assert originals[name] is ctx["components"][name]


def test_functools_wraps_replacement_is_rejected_by_object_identity(
    synthetic_factory,
):
    ctx = synthetic_factory()
    module = ctx["module"]
    original = ctx["components"]["sense_lidar"]

    @functools.wraps(original)
    def replacement(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **kwargs)

    assert replacement.__name__ == original.__name__
    assert replacement.__qualname__ == original.__qualname__
    assert replacement is not original

    module.sense_lidar = replacement

    with pytest.raises(
        profiler.ProfileError,
        match="sense_lidar is not the exact callable",
    ):
        profiler._original_callables(module)


def test_already_active_and_contaminated_guards_are_independent(
    synthetic_factory,
):
    ctx = synthetic_factory()
    module = ctx["module"]

    profiler._ACTIVE_MODULES.add(module)
    try:
        with pytest.raises(profiler.ProfileError, match="already active"):
            profiler._original_callables(module)
    finally:
        profiler._ACTIVE_MODULES.discard(module)

    profiler._CONTAMINATED_MODULES.add(module)
    try:
        with pytest.raises(profiler.ProfileError, match="contaminated"):
            profiler._original_callables(module)
    finally:
        profiler._CONTAMINATED_MODULES.discard(module)


def test_timed_wrapper_positive_control_counts_and_returns_original_result(
    monkeypatch: pytest.MonkeyPatch,
):
    readings = iter((1_000_000_000, 1_250_000_000))
    monkeypatch.setattr(profiler, "_clock_ns", lambda: next(readings))

    timer = profiler._empty_timer()
    marker = object()

    def original(value: Any, *, flag: bool = False) -> Any:
        assert flag is True
        return value

    wrapped = profiler._make_timed_wrapper(
        original=original,
        receipt=timer,
        label="positive-control",
    )

    assert wrapped(marker, flag=True) is marker
    assert timer["attempted_calls"] == 1
    assert timer["successful_calls"] == 1
    assert timer["inclusive_seconds"] == pytest.approx(0.25)
    assert timer["timing_valid"] is True
    assert timer["error"] is None
    assert getattr(wrapped, "__a22_profile_wrapper__") is True


def test_timed_wrapper_records_attempt_but_not_success_on_callable_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    readings = iter((100, 200))
    monkeypatch.setattr(profiler, "_clock_ns", lambda: next(readings))

    timer = profiler._empty_timer()
    expected = RuntimeError("original callable failed")

    def original() -> None:
        raise expected

    wrapped = profiler._make_timed_wrapper(
        original=original,
        receipt=timer,
        label="raising-control",
    )

    with pytest.raises(RuntimeError, match="original callable failed") as caught:
        wrapped()

    assert caught.value is expected
    assert timer["attempted_calls"] == 1
    assert timer["successful_calls"] == 0
    assert timer["inclusive_seconds"] > 0
    assert timer["timing_valid"] is True
    assert timer["error"] == "RuntimeError: original callable failed"


def test_timed_wrapper_start_clock_failure_does_not_call_original(
    monkeypatch: pytest.MonkeyPatch,
):
    timer = profiler._empty_timer()
    called = {"value": False}

    def broken_clock() -> int:
        raise RuntimeError("start clock failed")

    def original() -> None:
        called["value"] = True

    monkeypatch.setattr(profiler, "_clock_ns", broken_clock)

    wrapped = profiler._make_timed_wrapper(
        original=original,
        receipt=timer,
        label="start-clock-control",
    )

    with pytest.raises(RuntimeError, match="start clock failed"):
        wrapped()

    assert called["value"] is False
    assert timer["attempted_calls"] == 1
    assert timer["successful_calls"] == 0
    assert timer["timing_valid"] is False
    assert "start clock failed" in timer["error"]


def test_timed_wrapper_end_clock_failure_keeps_successful_call_count(
    monkeypatch: pytest.MonkeyPatch,
):
    readings = iter((100,))

    def clock() -> int:
        try:
            return next(readings)
        except StopIteration:
            raise RuntimeError("end clock failed")

    monkeypatch.setattr(profiler, "_clock_ns", clock)

    timer = profiler._empty_timer()
    called = {"value": False}

    def original() -> str:
        called["value"] = True
        return "returned"

    wrapped = profiler._make_timed_wrapper(
        original=original,
        receipt=timer,
        label="end-clock-control",
    )

    with pytest.raises(RuntimeError, match="end clock failed"):
        wrapped()

    assert called["value"] is True
    assert timer["attempted_calls"] == 1
    assert timer["successful_calls"] == 1
    assert timer["timing_valid"] is False
    assert "end clock failed" in timer["error"]


def test_end_clock_failure_does_not_replace_original_callable_exception(
    monkeypatch: pytest.MonkeyPatch,
):
    readings = iter((100,))

    def clock() -> int:
        try:
            return next(readings)
        except StopIteration:
            raise RuntimeError("secondary end clock failure")

    monkeypatch.setattr(profiler, "_clock_ns", clock)

    timer = profiler._empty_timer()
    original_error = ValueError("primary callable failure")

    def original() -> None:
        raise original_error

    wrapped = profiler._make_timed_wrapper(
        original=original,
        receipt=timer,
        label="exception-precedence-control",
    )

    with pytest.raises(ValueError, match="primary callable failure") as caught:
        wrapped()

    assert caught.value is original_error
    assert timer["attempted_calls"] == 1
    assert timer["successful_calls"] == 0
    assert timer["timing_valid"] is False
    assert "primary callable failure" in timer["error"]


def test_accumulator_rejects_overflow_without_committing_infinite_total(
    monkeypatch: pytest.MonkeyPatch,
):
    timer = profiler._empty_timer()
    timer["inclusive_seconds"] = 1.0e308

    # Exercise the accumulated-total overflow branch directly. A real
    # perf_counter_ns interval cannot conveniently approach this magnitude.
    monkeypatch.setattr(profiler, "_clock_ns", lambda: 1_000_000_000)
    monkeypatch.setattr(
        profiler,
        "_elapsed_seconds",
        lambda start, end, label: 1.0e308,
    )

    with pytest.raises(profiler.ProfileError, match="accumulated duration overflow"):
        profiler._accumulate_duration(timer, 0, "overflow-control")

    assert timer["inclusive_seconds"] == 1.0e308
    assert math.isfinite(timer["inclusive_seconds"])
    assert timer["timing_valid"] is False
    assert "overflow" in timer["error"]


def test_strict_json_sanitizes_nonfinite_values_and_rejects_profile(
):
    raw = {
        "status": "valid",
        "accepted_profile": True,
        "error": None,
        "timer": {
            "inclusive_seconds": float("inf"),
            "other": float("nan"),
        },
    }

    receipt = profiler._strict_json_receipt(raw)

    assert receipt["status"] == "invalid"
    assert receipt["accepted_profile"] is False
    assert receipt["timer"]["inclusive_seconds"] is None
    assert receipt["timer"]["other"] is None
    assert "non-finite float replaced with null" in receipt["error"]

    encoded = json.dumps(receipt, allow_nan=False)
    assert json.loads(encoded) == receipt


def test_artifact_inventory_accepts_all_required_structured_records():
    raw = {
        label: _hash_record(path)
        for label, path in BASE_ARTIFACT_PATHS.items()
    }

    normalized = profiler._normalize_artifact_inventory(
        raw,
        required_paths=BASE_ARTIFACT_PATHS,
    )

    assert normalized == {
        label: "a" * 64
        for label in sorted(BASE_ARTIFACT_PATHS)
    }


def test_artifact_inventory_rejects_missing_named_base_record():
    raw = {
        label: _hash_record(path)
        for label, path in BASE_ARTIFACT_PATHS.items()
        if label != "a22_tests"
    }

    with pytest.raises(
        profiler.ProfileError,
        match="a22_tests",
    ):
        profiler._normalize_artifact_inventory(
            raw,
            required_paths=BASE_ARTIFACT_PATHS,
        )


@pytest.mark.parametrize(
    ("record_change", "message"),
    [
        (lambda r: r.update(status="missing"), "verified status"),
        (lambda r: r.pop("status"), "verified status"),
        (lambda r: r.update(path=""), "no recorded path"),
        (lambda r: r.update(path="/absolute/file.py"), "invalid repository-relative"),
        (lambda r: r.update(path="../outside.py"), "invalid repository-relative"),
        (lambda r: r.update(path="benchmarks//runner.py"), "invalid repository-relative"),
        (lambda r: r.update(size_bytes=True), "invalid recorded file size"),
        (lambda r: r.update(size_bytes=-1), "invalid recorded file size"),
        (lambda r: r.update(size_bytes="123"), "invalid recorded file size"),
        (lambda r: r.update(sha256="not-a-digest"), "valid SHA-256"),
        (lambda r: r.update(sha256="A" * 64), "valid SHA-256"),
        (lambda r: r.pop("sha256"), "valid SHA-256"),
    ],
)
def test_artifact_inventory_rejects_malformed_record_contracts(
    record_change,
    message,
):
    record = _hash_record("benchmarks/environment_timing_a22.py")
    record_change(record)

    with pytest.raises(profiler.ProfileError, match=message):
        profiler._normalize_artifact_inventory(
            {"a22_runner": record},
        )


def test_artifact_inventory_rejects_record_path_not_matching_named_file():
    raw = {
        "a22_runner": _hash_record("benchmarks/not_the_runner.py"),
    }

    with pytest.raises(profiler.ProfileError, match="does not match"):
        profiler._normalize_artifact_inventory(
            raw,
            required_paths={"a22_runner": BASE_ARTIFACT_PATHS["a22_runner"]},
        )


def test_artifact_inventory_rejects_unstructured_legacy_digest_string():
    with pytest.raises(profiler.ProfileError, match="structured hash record"):
        profiler._normalize_artifact_inventory(
            {"a22_runner": "a" * 64},
        )


def test_artifact_hashes_requires_both_profile_evidence_files_and_valid_base_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    for relative in (
        "benchmarks/environment_profile_a22.py",
        "tests/test_environment_profile_a22.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")

    base_records = {
        label: _hash_record(path)
        for label, path in BASE_ARTIFACT_PATHS.items()
    }

    monkeypatch.setattr(
        profiler.timing,
        "_artifact_hashes",
        lambda root: base_records,
    )

    def file_hash(path: Path, *, root: Path) -> dict[str, Any]:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
        return _hash_record(relative, "b" * 64)

    monkeypatch.setattr(profiler.timing, "_sha256_file", file_hash)

    inventory = profiler._artifact_hashes(tmp_path)

    assert inventory["a21_benchmark"] == "a" * 64
    assert inventory["a22_runner"] == "a" * 64
    assert inventory["a22_tests"] == "a" * 64
    assert (
        inventory["benchmarks/environment_profile_a22.py"]
        == "b" * 64
    )
    assert (
        inventory["tests/test_environment_profile_a22.py"]
        == "b" * 64
    )


def test_artifact_hashes_rejects_missing_profile_evidence_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    profile_file = tmp_path / "benchmarks/environment_profile_a22.py"
    profile_file.parent.mkdir(parents=True, exist_ok=True)
    profile_file.write_text("# fixture\n", encoding="utf-8")

    monkeypatch.setattr(
        profiler.timing,
        "_artifact_hashes",
        lambda root: {
            label: _hash_record(path)
            for label, path in BASE_ARTIFACT_PATHS.items()
        },
    )

    with pytest.raises(profiler.ProfileError, match="test_environment_profile_a22.py"):
        profiler._artifact_hashes(tmp_path)


def test_artifact_hashes_rejects_missing_base_key_even_with_other_valid_records(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    for relative in (
        "benchmarks/environment_profile_a22.py",
        "tests/test_environment_profile_a22.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")

    incomplete = {
        label: _hash_record(path)
        for label, path in BASE_ARTIFACT_PATHS.items()
        if label != "a22_tests"
    }
    monkeypatch.setattr(
        profiler.timing,
        "_artifact_hashes",
        lambda root: incomplete,
    )

    with pytest.raises(profiler.ProfileError, match="a22_tests"):
        profiler._artifact_hashes(tmp_path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"batch": 0}, "batch"),
        ({"candidates": 0}, "candidates"),
        ({"batch": 2}, "B1/N1"),
        ({"candidates": 2}, "B1/N1"),
    ],
)
def test_profile_rejects_invalid_arguments_before_source_or_install(
    synthetic_factory,
    kwargs,
    message,
):
    ctx = synthetic_factory()

    arguments = {
        "seed": 7,
        "batch": 1,
        "candidates": 1,
        "output_path": ctx["output"],
    }
    arguments.update(kwargs)

    receipt = profiler.profile_episode(**arguments)

    assert receipt["status"] == "invalid"
    assert receipt["accepted_profile"] is False
    assert receipt["error"] is not None
    assert message.lower() in receipt["error"].lower()
    assert receipt["validation"]["source_before"] == "not_run"
    assert receipt["validation"]["artifacts_before"] == "not_run"
    assert receipt["restoration"]["status"] == "not_attempted"

    # Argument rejection occurs before any synthetic module is hooked.
    assert ctx["module"] not in profiler._ACTIVE_MODULES
    assert ctx["module"] not in profiler._CONTAMINATED_MODULES
    assert ctx["module"].step is ctx["public_step"]
    assert profiler.A21.step is ctx["public_step"]


def test_profile_seed_zero_is_valid_and_runs_complete_synthetic_episode(
    synthetic_factory,
):
    ctx = synthetic_factory()

    receipt = profiler.profile_episode(
        seed=0,
        batch=1,
        candidates=1,
        output_path=ctx["output"],
    )

    assert receipt["workload"]["seed"] == 0
    assert receipt["validation"]["arguments"] == "passed"
    assert receipt["progress"]["transitions_attempted"] == 2400
    assert receipt["progress"]["public_step_calls_returned"] == 2400
    assert receipt["accepted_profile"] is True


def test_source_change_during_profile_invalidates_acceptance(
    synthetic_factory,
):
    ctx = synthetic_factory(source_change=True)

    receipt = profiler.profile_episode(
        seed=7,
        batch=1,
        candidates=1,
        output_path=ctx["output"],
    )

    assert receipt["source_before"]["fingerprint"] == "source-before"
    assert receipt["source_after"]["fingerprint"] == "source-after"
    assert receipt["source_unchanged"] is False
    assert receipt["validation"]["source_after"] == "failed"
    assert receipt["status"] == "invalid"
    assert receipt["accepted_profile"] is False
    assert "source identity changed" in receipt["error"]
    _assert_synthetic_identities_restored(ctx)


def test_artifact_change_during_profile_invalidates_acceptance(
    synthetic_factory,
):
    ctx = synthetic_factory(artifact_change=True)

    receipt = profiler.profile_episode(
        seed=7,
        batch=1,
        candidates=1,
        output_path=ctx["output"],
    )

    assert receipt["artifacts_before"] == {"synthetic_artifact": "a" * 64}
    assert receipt["artifacts_after"] == {"synthetic_artifact": "b" * 64}
    assert receipt["artifacts_unchanged"] is False
    assert receipt["validation"]["artifacts_after"] == "failed"
    assert receipt["status"] == "invalid"
    assert receipt["accepted_profile"] is False
    assert "artifact hashes changed" in receipt["error"]
    _assert_synthetic_identities_restored(ctx)


def test_real_unmocked_seed7_b1_n1_profile_persists_receipt_and_restores_production(
    tmp_path: Path,
):
    """One actual CPU episode through the unmodified production call graph."""

    step_module = importlib.import_module("lawless.environment.step")

    owners = {
        name: importlib.import_module(module_name)
        for name, module_name in profiler.OWNER_MODULES.items()
    }
    owner_names = {
        "physics_step": "step",
        "resolve_static_obstacles": "resolve_static_obstacles",
        "sense_lidar": "sense_lidar",
        "assemble_observations": "assemble_observations",
    }

    original_components = {
        name: getattr(step_module, name)
        for name in profiler.COMPONENT_NAMES
    }
    original_module_step = step_module.step
    original_a21_step = profiler.A21.step

    # Establish the actual production identity relationships before running.
    assert original_module_step is original_a21_step
    for name in profiler.COMPONENT_NAMES:
        assert original_components[name] is getattr(
            owners[name],
            owner_names[name],
        )

    output = tmp_path / "actual-a22-cpu-profile.json"

    # No production functions, owner functions, A21 methods, source identity,
    # artifact hashing, or timing helpers are mocked in this smoke test.
    receipt = profiler.profile_episode(
        seed=7,
        batch=1,
        candidates=1,
        output_path=output,
    )

    assert receipt["status"] == "valid"
    assert receipt["accepted_profile"] is True
    assert receipt["workload"]["seed"] == 7
    assert receipt["workload"]["batch"] == 1
    assert receipt["workload"]["candidates"] == 1
    assert receipt["workload"]["device"] == "cpu"
    assert receipt["workload"]["dtype"] == "torch.float32"
    assert receipt["workload"]["episode_steps"] == 2400
    assert receipt["workload"]["warmup_repetitions"] == 0
    assert receipt["workload"]["profiled_repetitions"] == 1

    assert receipt["progress"]["transitions_attempted"] == 2400
    assert receipt["progress"]["public_step_calls_attempted"] == 2400
    assert receipt["progress"]["public_step_calls_returned"] == 2400

    assert receipt["public_step"]["attempted_calls"] == 2400
    assert receipt["public_step"]["successful_calls"] == 2400
    assert receipt["public_step"]["timing_valid"] is True
    assert receipt["public_step"]["inclusive_seconds"] > 0

    for name in profiler.COMPONENT_NAMES:
        timer = receipt["components"][name]
        assert timer["attempted_calls"] == 2400
        assert timer["successful_calls"] == 2400
        assert timer["timing_valid"] is True
        assert timer["error"] is None
        assert timer["inclusive_seconds"] > 0
        assert (
            receipt["public_step"]["inclusive_seconds"]
            >= timer["inclusive_seconds"]
        )

    assert receipt["validation"]["arguments"] == "passed"
    assert receipt["validation"]["source_before"] == "verified"
    assert receipt["validation"]["artifacts_before"] == "verified"
    assert receipt["validation"]["initial_environment"] == "passed"
    assert receipt["validation"]["reset_observations"] == "passed"
    assert receipt["validation"]["final_observations"] == "passed"
    assert receipt["validation"]["final_environment"] == "passed"
    assert receipt["validation"]["component_attribution"] == "passed"
    assert receipt["validation"]["restoration"] == "verified"
    assert receipt["validation"]["source_after"] == "verified"
    assert receipt["validation"]["artifacts_after"] == "verified"

    assert receipt["source_unchanged"] is True
    assert receipt["artifacts_unchanged"] is True
    assert receipt["restoration"]["status"] == "verified"
    assert all(receipt["restoration"]["identities"].values())
    assert receipt["error"] is None

    # Prove the real imported component functions and both public aliases
    # have been restored to the exact pre-profile objects.
    assert step_module.step is original_module_step
    assert profiler.A21.step is original_a21_step
    assert step_module.step is profiler.A21.step

    for name, original in original_components.items():
        assert getattr(step_module, name) is original
        assert getattr(step_module, name) is getattr(
            owners[name],
            owner_names[name],
        )

    assert step_module not in profiler._ACTIVE_MODULES
    assert step_module not in profiler._CONTAMINATED_MODULES

    # The file is the actual raw receipt returned by the profiler, not a
    # reconstructed or normalized summary.
    assert output.is_file()
    persisted = _strict_load(output)
    assert persisted == receipt
    json.dumps(persisted, sort_keys=True, allow_nan=False)

    assert persisted["schema"] == "lawless.a22.component_profile.v2"
    assert persisted["status"] == "valid"
    assert persisted["accepted_profile"] is True
