
#!/usr/bin/env python3
"""
LAWLESS P1.8 B16/N256 CUDA memory + tensor-shape probe.

Run from Colab:
    %run /content/lawless_b16n256_cuda_memory_probe.py

The script must live OUTSIDE the pinned repository.
It never writes into the repository.
"""
from __future__ import annotations

import ast
import base64
import dataclasses
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

PIN = "0a6b409e9433a428cb6b34e02f0cd96c7cb85b6a"
CLONE = Path("/content/LAWLESS_cuda_smoke_0a6b409e9433").resolve()
B = 16
N = 256
SEED = 7
TIMEOUT_SECONDS = 300
EXPECTED_OBS = (16, 256, 23)
EXPECTED_ACTIONS = (16, 256, 2)
EXPECTED_AGENT_STATE = (N, 2)
EXPECTED_AGENT_VECTOR = (N,)
EXPECTED_WORLD_AGENT = (B, N)
FACTORY_RELATIVE_PATH = "benchmarks/environment_benchmark.py"

FACTORY_SOURCE = """\
def create_config(seed: int) -> EnvironmentConfig:
    config = EnvironmentConfig()
    config.seed = seed
    config.validate()
    return config
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=CLONE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    return result.stdout.strip()


def tracked_python_files() -> list[Path]:
    names = run_git("ls-files", "-z").split("\0")
    return [
        CLONE / name
        for name in names
        if name.endswith(".py") and (CLONE / name).is_file()
    ]


def source_manifest() -> dict:
    """Hash tracked LAWLESS production sources and the exact benchmark factory."""
    production = {}
    factory_matches = []

    expected_ast = ast.dump(
        ast.parse(FACTORY_SOURCE).body[0],
        include_attributes=False,
    )

    for path in tracked_python_files():
        relative = path.relative_to(CLONE).as_posix()

        if relative.startswith("lawless/"):
            production[relative] = sha256_file(path)

        if relative != FACTORY_RELATIVE_PATH:
            continue

        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and node.name == "create_config"
                and ast.dump(node, include_attributes=False) == expected_ast
            ):
                factory_matches.append({
                    "path": relative,
                    "sha256": sha256_file(path),
                })

    if not production:
        raise RuntimeError("No tracked lawless/*.py production sources found")

    if len(factory_matches) != 1:
        raise RuntimeError(
            "Expected exactly one unchanged create_config(seed) factory "
            f"in {FACTORY_RELATIVE_PATH}; found {len(factory_matches)}"
        )

    return {
        "production_python_sha256": production,
        "benchmark_factory_matches": factory_matches,
    }


def checkout_status() -> dict:
    return {
        "head": run_git("rev-parse", "HEAD"),
        "status_porcelain": run_git(
            "status", "--porcelain", "--untracked-files=all"
        ),
    }


def verify_checkout() -> dict:
    state = checkout_status()
    if state["head"] != PIN:
        raise RuntimeError(
            f"Wrong checkout: expected {PIN}, got {state['head']}"
        )
    if state["status_porcelain"]:
        raise RuntimeError(
            "Checkout is not clean:\n" + state["status_porcelain"]
        )
    return state


def finite_status(tensor):
    import torch

    if tensor.is_floating_point() or tensor.is_complex():
        return bool(torch.isfinite(tensor).all().item())
    return None


def tensor_audit(root, root_name: str) -> list[dict]:
    """
    Recursively inspect tensors in dataclasses, mappings, sequences,
    and ordinary object attributes.

    Called AFTER allocator peak measurements. Tensor aliases are reported
    at each path but traversed only once.
    """
    import torch

    records = []
    seen_objects = set()
    seen_tensors = {}

    def walk(value, path):
        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            record = {
                "path": path,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "finite": finite_status(value),
                "numel": value.numel(),
                "storage_bytes": (
                    value.untyped_storage().nbytes()
                    if value.device.type != "meta"
                    else None
                ),
            }
            if tensor_id in seen_tensors:
                record["alias_of"] = seen_tensors[tensor_id]
            else:
                seen_tensors[tensor_id] = path
            records.append(record)
            return

        if value is None or isinstance(
            value, (str, bytes, int, float, bool, type, Path)
        ):
            return

        oid = id(value)
        if oid in seen_objects:
            return
        seen_objects.add(oid)

        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                walk(getattr(value, field.name), f"{path}.{field.name}")
            return

        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, f"{path}[{key!r}]")
            return

        if isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")
            return

        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, dict):
            for key, child in attributes.items():
                walk(child, f"{path}.{key}")

    walk(root, root_name)
    return records


def assert_all_finite(records, label):
    bad = [
        item["path"]
        for item in records
        if item["finite"] is False
    ]
    if bad:
        raise AssertionError(
            f"{label}: non-finite tensors found: {bad[:30]}"
        )


def allocator_snapshot(torch, label):
    torch.cuda.synchronize()
    return {
        "label": label,
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def assert_tensor_contract(
    name,
    tensor,
    expected_shape,
    expected_dtype,
    expected_device,
):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(
            f"{name}: expected Tensor, got {type(tensor).__name__}"
        )
    if tuple(tensor.shape) != tuple(expected_shape):
        raise AssertionError(
            f"{name}: shape {tuple(tensor.shape)} != "
            f"expected {tuple(expected_shape)}"
        )
    if tensor.dtype != expected_dtype:
        raise AssertionError(
            f"{name}: dtype {tensor.dtype} != expected {expected_dtype}"
        )
    if tensor.device != expected_device:
        raise AssertionError(
            f"{name}: device {tensor.device} != expected {expected_device}"
        )
    if tensor.is_floating_point() or tensor.is_complex():
        if not bool(torch.isfinite(tensor).all().item()):
            raise AssertionError(f"{name}: contains non-finite values")


def load_actual_benchmark_factory():
    """
    Compile only the exact create_config function from the required
    tracked benchmark file. This executes its actual source unchanged
    without importing and running the benchmark module's other top-level
    code.
    """
    import torch
    from lawless.world.config import EnvironmentConfig

    factory_path = CLONE / FACTORY_RELATIVE_PATH
    if not factory_path.is_file():
        raise RuntimeError(f"Benchmark factory file missing: {factory_path}")

    source = factory_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(factory_path))
    expected_ast = ast.dump(
        ast.parse(FACTORY_SOURCE).body[0],
        include_attributes=False,
    )

    matches = [
        node
        for node in ast.walk(tree)
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "create_config"
            and ast.dump(node, include_attributes=False) == expected_ast
        )
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Benchmark factory source no longer matches the supplied "
            f"create_config implementation; matches={len(matches)}"
        )

    module = ast.Module(body=[matches[0]], type_ignores=[])
    code = compile(module, str(factory_path), "exec")
    namespace = {"EnvironmentConfig": EnvironmentConfig}
    exec(code, namespace)

    create_config = namespace["create_config"]
    config = create_config(SEED)

    if config.seed != SEED:
        raise AssertionError("Actual benchmark factory seed mismatch")

    return config, {
        "path": FACTORY_RELATIVE_PATH,
        "sha256": sha256_file(factory_path),
        "factory_source": FACTORY_SOURCE,
    }


def child_probe(receipt_path: Path):
    """CUDA work only. Parent enforces the hard wall-clock timeout."""
    receipt = {
        "status": "started",
        "seed": SEED,
        "worlds": B,
        "agents_per_world": N,
        "candidate_slots": B * N,
        "dtype_requested": "torch.float32",
        "device_requested": "cuda",
        "public_transitions_requested": 2,
        "measurement_scope": (
            "Baseline is sampled before public reset. Reset peak is sampled "
            "immediately after reset and before recursive reset audit. "
            "Step peak is reset after reset-audit cleanup and sampled "
            "immediately after two public steps, before final audit. "
            "The actions tensor is allocated after the step peak reset and "
            "is intentionally included in the step-phase peak window. "
            "Step reserved peak may include cached allocator blocks retained "
            "from reset and its audit; it is not isolated incremental "
            "step-only reserved-memory demand."
        ),
    }

    try:
        import torch
        from lawless.environment import reset, step

        receipt["python"] = sys.version
        receipt["platform"] = platform.platform()
        receipt["torch"] = torch.__version__
        receipt["cuda_runtime"] = torch.version.cuda
        receipt["cuda_available"] = torch.cuda.is_available()

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable; no CPU fallback permitted"
            )

        torch.cuda.init()
        device = torch.device("cuda:0")
        receipt["cuda_device"] = torch.cuda.get_device_name(device)
        receipt["cuda_device_index"] = torch.cuda.current_device()

        reset_path = Path(inspect.getfile(reset)).resolve()
        step_path = Path(inspect.getfile(step)).resolve()
        receipt["import_paths"] = {
            "reset": str(reset_path),
            "step": str(step_path),
        }

        for name, path in (
            ("reset", reset_path),
            ("step", step_path),
        ):
            if not path.is_relative_to(CLONE):
                raise RuntimeError(
                    f"{name} imported outside pinned clone: {path}"
                )

        config, factory_record = load_actual_benchmark_factory()
        receipt["benchmark_factory"] = factory_record

        torch.cuda.synchronize()
        receipt["allocator_baseline"] = {
            "label": "before_reset",
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
        }

        torch.cuda.reset_peak_memory_stats(device)

        # Actual public reset.
        environment, observations, reset_info = reset(
            config,
            num_worlds=B,
            num_agents=N,
            device=device,
            dtype=torch.float32,
        )

        # Capture reset allocator values before assertions or recursive audit.
        receipt["post_reset_allocator"] = allocator_snapshot(
            torch, "post_reset_before_audit_and_contract_checks"
        )
        receipt["reset_peak"] = {
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        }

        # Explicit reset contract checks occur outside the reset peak window.
        if len(environment.worlds) != B:
            raise AssertionError(
                f"Reset returned {len(environment.worlds)} worlds; expected {B}"
            )

        assert_tensor_contract(
            "reset.observations",
            observations,
            EXPECTED_OBS,
            torch.float32,
            device,
        )

        layout_ids = []
        spawn_shapes = []
        generated_spawn_shapes = []

        for world_index, world in enumerate(environment.worlds):
            episode = world.episode
            physics = episode.physics

            assert_tensor_contract(
                f"reset.world[{world_index}].physics.position",
                physics.position,
                EXPECTED_AGENT_STATE,
                torch.float32,
                device,
            )
            assert_tensor_contract(
                f"reset.world[{world_index}].spawn_indices",
                episode.spawn_indices,
                EXPECTED_AGENT_VECTOR,
                torch.long,
                device,
            )

            generated_world = episode.generated_world
            assert_tensor_contract(
                f"reset.world[{world_index}].generated_world.spawn_positions",
                generated_world.spawn_positions,
                (1, 2),
                torch.float32,
                device,
            )

            if not bool((episode.spawn_indices == 0).all().item()):
                raise AssertionError(
                    f"world[{world_index}]: expected all candidate spawn "
                    "indices to reference the single generated spawn"
                )

            layout_ids.append(id(generated_world))
            spawn_shapes.append(list(episode.spawn_indices.shape))
            generated_spawn_shapes.append(
                list(generated_world.spawn_positions.shape)
            )

        if len(set(layout_ids)) != B:
            raise AssertionError(
                "Expected one distinct generated layout object per world"
            )

        receipt["geometry_contract"] = {
            "world_count": len(environment.worlds),
            "generated_world_objects": len(layout_ids),
            "generated_world_object_ids": layout_ids,
            "distinct_layout_objects_across_worlds": (
                len(set(layout_ids)) == B
            ),
            "generated_spawn_shapes": generated_spawn_shapes,
            "spawn_index_shapes": spawn_shapes,
            "all_spawn_indices_zero": True,
            "layout_sharing_semantics": (
                "Each world's AgentEpisodeState references its single "
                "GeneratedWorld object; its N candidate agents use spawn "
                "index zero. Distinct GeneratedWorld objects are checked "
                "across the B worlds."
            ),
        }

        reset_audit = tensor_audit(
            {
                "environment": environment,
                "observations": observations,
                "reset_info": reset_info,
            },
            "reset",
        )
        assert_all_finite(reset_audit, "reset")
        receipt["reset_tensor_audit"] = reset_audit

        del reset_audit
        import gc
        gc.collect()
        torch.cuda.synchronize()

        # New peak window for two public transitions. Actions allocation is
        # deliberately inside this window and included in the measured peak.
        torch.cuda.reset_peak_memory_stats(device)

        actions = torch.zeros(
            EXPECTED_ACTIONS,
            dtype=torch.float32,
            device=device,
        )

        receipt["actions"] = {
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "device": str(actions.device),
            "candidate_slots": B * N,
            "included_in_step_peak_window": True,
        }

        step_records = []

        for index in range(2):
            result = step(environment, actions)

            if not isinstance(result, tuple) or len(result) != 6:
                raise AssertionError(
                    "Public step did not return the documented six-tuple"
                )

            (
                environment,
                observations,
                rewards,
                terminated,
                truncated,
                info,
            ) = result

            torch.cuda.synchronize()

            step_records.append({
                "transition": index + 1,
                "observations_shape": list(observations.shape),
                "rewards_shape": list(rewards.shape),
                "terminated_shape": list(terminated.shape),
                "truncated_shape": list(truncated.shape),
                "step_counts": [
                    int(world.step_count) for world in environment.worlds
                ],
            })

        # Capture raw allocator values immediately after both public steps,
        # before contract assertions and recursive final tensor audit.
        receipt["post_two_steps_allocator"] = allocator_snapshot(
            torch, "after_two_public_steps_before_contract_checks_and_audit"
        )
        receipt["step_phase_peak"] = {
            "peak_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        }

        receipt["step_phase_peak_scope"] = (
            "The step-phase peak window begins after reset-audit cleanup. "
            "Actions allocation is included. The reserved peak may include "
            "cached allocator blocks retained from reset and its audit; "
            "it is not isolated incremental step-only reserved-memory demand."
        )

        # Explicit step contract checks occur outside the step peak window.
        if len(environment.worlds) != B:
            raise AssertionError(
                f"Step returned {len(environment.worlds)} worlds; expected {B}"
            )

        assert_tensor_contract(
            "step.observations",
            observations,
            EXPECTED_OBS,
            torch.float32,
            device,
        )
        assert_tensor_contract(
            "step.rewards",
            rewards,
            EXPECTED_WORLD_AGENT,
            torch.float32,
            device,
        )
        assert_tensor_contract(
            "step.terminated",
            terminated,
            EXPECTED_WORLD_AGENT,
            torch.bool,
            device,
        )
        assert_tensor_contract(
            "step.truncated",
            truncated,
            EXPECTED_WORLD_AGENT,
            torch.bool,
            device,
        )

        for world_index, world in enumerate(environment.worlds):
            if world.step_count != 2:
                raise AssertionError(
                    f"world[{world_index}] step_count={world.step_count}; "
                    "expected 2"
                )

            assert_tensor_contract(
                f"step.world[{world_index}].physics.position",
                world.episode.physics.position,
                EXPECTED_AGENT_STATE,
                torch.float32,
                device,
            )
            assert_tensor_contract(
                f"step.world[{world_index}].spawn_indices",
                world.episode.spawn_indices,
                EXPECTED_AGENT_VECTOR,
                torch.long,
                device,
            )

        receipt["step_records"] = step_records
        receipt["final_step_counts"] = [
            int(world.step_count) for world in environment.worlds
        ]
        receipt["public_step_calls_completed"] = 2

        # Recursive final audit is after both allocator peak captures.
        final_audit = tensor_audit(
            {
                "environment": environment,
                "observations": observations,
                "actions": actions,
                "rewards": rewards,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            },
            "after_two_steps",
        )
        assert_all_finite(final_audit, "after_two_steps")
        receipt["final_tensor_audit"] = final_audit

        receipt["audit_summary"] = {
            "reset_tensor_paths": len(receipt["reset_tensor_audit"]),
            "final_tensor_paths": len(final_audit),
            "reset_unique_tensor_objects": len({
                item["path"]
                for item in receipt["reset_tensor_audit"]
                if "alias_of" not in item
            }),
            "final_unique_tensor_objects": len({
                item["path"]
                for item in final_audit
                if "alias_of" not in item
            }),
            "all_audited_tensors_finite": True,
        }

        receipt["status"] = "passed"

    except BaseException as exc:
        torch_module = sys.modules.get("torch")
        cuda_module = (
            getattr(torch_module, "cuda", None)
            if torch_module is not None
            else None
        )
        oom_type = (
            getattr(cuda_module, "OutOfMemoryError", None)
            if cuda_module is not None
            else None
        )

        is_cuda_oom = (
            isinstance(oom_type, type)
            and isinstance(exc, oom_type)
        )

        receipt["status"] = "cuda_oom" if is_cuda_oom else "failed"
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
        receipt["traceback"] = traceback.format_exc()

        if is_cuda_oom:
            try:
                if (
                    torch_module is not None
                    and torch_module.cuda.is_available()
                ):
                    torch_module.cuda.synchronize()
                    receipt["oom_allocator_snapshot"] = {
                        "allocated_bytes": int(
                            torch_module.cuda.memory_allocated()
                        ),
                        "reserved_bytes": int(
                            torch_module.cuda.memory_reserved()
                        ),
                    }
            except BaseException as snapshot_error:
                receipt["oom_snapshot_error"] = (
                    f"{type(snapshot_error).__name__}: {snapshot_error}"
                )

    finally:
        receipt["finished_unix"] = time.time()
        receipt_path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    return 0 if receipt["status"] == "passed" else 1


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        return child_probe(Path(sys.argv[2]))

    script_path = Path(__file__).resolve()
    script_bytes = script_path.read_bytes()
    script_hash = sha256_bytes(script_bytes)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    evidence_dir = Path(tempfile.mkdtemp(
        prefix=f"LAWLESS_B16N256_{run_id}_",
        dir="/content",
    ))

    child_receipt_path = evidence_dir / "child_receipt.json"
    stdout_path = evidence_dir / "child_stdout.txt"
    stderr_path = evidence_dir / "child_stderr.txt"
    parent_receipt_path = evidence_dir / "parent_receipt.json"

    parent_receipt = {
        "run_id": run_id,
        "probe_script": script_path.name,
        "probe_script_sha256": script_hash,
        "clone": str(CLONE),
        "expected_commit": PIN,
        "timeout_seconds": TIMEOUT_SECONDS,
        "started_unix": time.time(),
        "status": "started",
    }

    pre_manifest = None
    pre_checkout = None
    child_returncode = None
    child_receipt = None
    postflight_passed = False

    try:
        # Keep this guard inside the parent try so failures still produce
        # parent receipts and the recovery ZIP.
        if script_path == CLONE or CLONE in script_path.parents:
            raise RuntimeError(
                "Probe script must be stored outside the pinned repository."
            )

        if not CLONE.is_dir():
            raise RuntimeError(f"Pinned clone not found: {CLONE}")

        pre_checkout = verify_checkout()
        pre_manifest = source_manifest()

        parent_receipt["pre_checkout"] = pre_checkout
        parent_receipt["pre_source_manifest"] = pre_manifest

        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = (
            str(CLONE)
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        command = [
            sys.executable,
            str(script_path),
            "--child",
            str(child_receipt_path),
        ]

        try:
            completed = subprocess.run(
                command,
                cwd=CLONE,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )

            child_returncode = completed.returncode
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")

            parent_receipt["child_returncode"] = child_returncode
            parent_receipt["child_stdout_bytes"] = len(
                completed.stdout.encode("utf-8")
            )
            parent_receipt["child_stderr_bytes"] = len(
                completed.stderr.encode("utf-8")
            )

        except subprocess.TimeoutExpired as exc:
            parent_receipt["status"] = "timeout"
            parent_receipt["timeout_seconds"] = TIMEOUT_SECONDS

            stdout = exc.stdout or ""
            stderr = exc.stderr or ""

            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")

            stdout_path.write_text(stdout, encoding="utf-8")
            stderr_path.write_text(stderr, encoding="utf-8")

            parent_receipt["timeout_stdout"] = stdout[-12000:]
            parent_receipt["timeout_stderr"] = stderr[-12000:]

        if child_receipt_path.exists():
            child_receipt = json.loads(
                child_receipt_path.read_text(encoding="utf-8")
            )
            parent_receipt["child_status"] = child_receipt.get("status")
            parent_receipt["child_public_step_calls_completed"] = (
                child_receipt.get("public_step_calls_completed")
            )
        else:
            parent_receipt["child_status"] = "no_receipt"

    except BaseException as exc:
        parent_receipt["status"] = "parent_failed"
        parent_receipt["error_type"] = type(exc).__name__
        parent_receipt["error"] = str(exc)
        parent_receipt["traceback"] = traceback.format_exc()

    finally:
        # Always attempt both postflight checks, including after timeout,
        # child failure, source-guard failure, or parent exception.
        post_checkout = None
        post_manifest = None
        postflight_errors = []

        try:
            post_checkout = verify_checkout()
            parent_receipt["post_checkout"] = post_checkout
        except BaseException as exc:
            parent_receipt["post_checkout_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            postflight_errors.append("post_checkout failed")

        try:
            post_manifest = source_manifest()
            parent_receipt["post_source_manifest"] = post_manifest
        except BaseException as exc:
            parent_receipt["post_manifest_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            postflight_errors.append("post_source_manifest failed")

        checkout_unchanged = (
            pre_checkout is not None
            and post_checkout is not None
            and pre_checkout == post_checkout
            and post_checkout.get("head") == PIN
            and post_checkout.get("status_porcelain") == ""
        )

        production_source_unchanged = (
            pre_manifest is not None
            and post_manifest is not None
            and pre_manifest.get("production_python_sha256")
            == post_manifest.get("production_python_sha256")
        )

        benchmark_factory_source_unchanged = (
            pre_manifest is not None
            and post_manifest is not None
            and pre_manifest.get("benchmark_factory_matches")
            == post_manifest.get("benchmark_factory_matches")
        )

        postflight_passed = (
            not postflight_errors
            and checkout_unchanged
            and production_source_unchanged
            and benchmark_factory_source_unchanged
        )

        parent_receipt["checkout_unchanged"] = checkout_unchanged
        parent_receipt["production_source_unchanged"] = (
            production_source_unchanged
        )
        parent_receipt["benchmark_factory_source_unchanged"] = (
            benchmark_factory_source_unchanged
        )
        parent_receipt["postflight_passed"] = postflight_passed
        parent_receipt["postflight_errors"] = postflight_errors

        child_passed = (
            child_returncode == 0
            and child_receipt is not None
            and child_receipt.get("status") == "passed"
            and child_receipt.get("public_step_calls_completed") == 2
        )

        parent_receipt["child_passed"] = child_passed

        # Fail closed: no status can remain "completed" unless the child
        # receipt proves both public calls and every postflight check passed.
        if not postflight_passed:
            parent_receipt["status"] = "postflight_failed"
        elif child_passed:
            parent_receipt["status"] = "completed"
        elif parent_receipt.get("status") == "timeout":
            parent_receipt["status"] = "timeout"
        elif child_receipt is not None:
            child_status = child_receipt.get("status")
            parent_receipt["status"] = (
                child_status
                if child_status not in (None, "started", "passed")
                else "child_failed"
            )
        elif parent_receipt.get("status") == "parent_failed":
            parent_receipt["status"] = "parent_failed"
        else:
            parent_receipt["status"] = "child_failed_no_receipt"

        parent_receipt["finished_unix"] = time.time()

        parent_receipt_path.write_text(
            json.dumps(parent_receipt, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        # Include exact probe bytes and every available receipt/output file.
        archive_path = Path("/content") / (
            f"LAWLESS_B16N256_cuda_memory_{run_id}_{os.getpid()}.zip"
        )

        try:
            with zipfile.ZipFile(
                archive_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
            ) as archive:
                archive.writestr(script_path.name, script_bytes)
                for path in sorted(evidence_dir.iterdir()):
                    if path.is_file():
                        archive.write(path, arcname=path.name)

            archive_bytes = archive_path.read_bytes()
            archive_hash = sha256_bytes(archive_bytes)

            print("\n=== LAWLESS B16/N256 PROBE RECEIPT ===")
            print(json.dumps(parent_receipt, indent=2, sort_keys=True))

            if child_receipt_path.exists():
                print("\n=== CHILD CUDA RECEIPT ===")
                print(child_receipt_path.read_text(encoding="utf-8"))
            else:
                print("\nNo child receipt was produced.")

            print("\n=== EVIDENCE ZIP ===")
            print(f"path={archive_path}")
            print(f"bytes={len(archive_bytes)}")
            print(f"sha256={archive_hash}")

            # ZIP's own hash is deliberately not written inside the ZIP.
            print("\n=== ZIP BASE64 RECOVERY ===")
            encoded = base64.b64encode(archive_bytes).decode("ascii")
            print("ZIP_BASE64_BEGIN")
            print(encoded)
            print("ZIP_BASE64_END")

        except BaseException as archive_error:
            print("\n=== EVIDENCE ARCHIVE FAILURE ===")
            print(
                f"{type(archive_error).__name__}: {archive_error}"
            )
            print(
                "Parent and child receipts, if written, remain in: "
                f"{evidence_dir}"
            )

    return 0 if parent_receipt.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
