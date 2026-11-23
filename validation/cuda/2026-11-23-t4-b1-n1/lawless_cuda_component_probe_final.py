# Parent/CLI/imports/constants. Append after the retained utility and child functions.

import argparse
import base64
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import torch

ROOT = "/content/LAWLESS_cuda_smoke_0a6b409e9433"
EXPECTED_COMMIT = "0a6b409e9433a428cb6b34e02f0cd96c7cb85b6a"
PARENT_BUDGET = 150
CHILD_BUDGET = 120


def git_guard(root):
    """Require the exact pinned, clean checkout."""
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    head = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--untracked-files=all")
    if head != EXPECTED_COMMIT:
        raise RuntimeError(f"Unexpected HEAD: {head}")
    if status:
        raise RuntimeError(f"Dirty checkout:\n{status}")
    return head


def source_manifest(root):
    """SHA256 inventory of tracked Python files."""
    paths = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True, capture_output=True,
    ).stdout.decode().split("\0")
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in paths
        if name.endswith(".py") and (root / name).is_file()
    }


def save_json(path, obj):
    """Persist partial or final probe evidence atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)

def tensor_audit(value):
    """Recursively audit tensors, rejecting non-CUDA or nonfinite float tensors."""
    import dataclasses

    records = []
    seen = set()

    def visit(obj, path):
        identity = id(obj)
        if identity in seen:
            return
        seen.add(identity)

        if isinstance(obj, torch.Tensor):
            record = {
                "path": path,
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
            }
            if obj.device.type != "cuda":
                raise RuntimeError(f"Non-CUDA tensor at {path}: {obj.device}")
            if obj.is_floating_point():
                finite = bool(torch.isfinite(obj).all().item())
                record["finite"] = finite
                if not finite:
                    raise RuntimeError(f"Nonfinite floating tensor at {path}")
            else:
                record["finite"] = None
            records.append(record)
            return

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            for field in dataclasses.fields(obj):
                visit(getattr(obj, field.name), f"{path}.{field.name}")
        elif isinstance(obj, dict):
            for key, child in obj.items():
                visit(child, f"{path}[{key!r}]")
        elif isinstance(obj, (tuple, list)):
            for index, child in enumerate(obj):
                visit(child, f"{path}[{index}]")

    visit(value, "root")
    return records


def timed_cuda_call(torch, fn, repeats):
    """Measure synchronized CUDA-call wall time and memory per repeat."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")

    samples = []
    last_value = None

    for _ in range(repeats):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()

        start = time.perf_counter()
        last_value = fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        samples.append({
            "seconds": elapsed,
            "allocated_before": allocated_before,
            "reserved_before": reserved_before,
            "allocated_after": torch.cuda.memory_allocated(),
            "reserved_after": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
        })

    return last_value, samples

def assert_tensor_contract(tensor, shape, dtype):
    """Enforce exact shape and dtype; CUDA/finiteness remain covered by audit."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected tensor, got {type(tensor).__name__}")
    if tuple(tensor.shape) != tuple(shape):
        raise AssertionError(f"Shape {tuple(tensor.shape)} != {tuple(shape)}")
    if tensor.dtype != dtype:
        raise AssertionError(f"Dtype {tensor.dtype} != {dtype}")

def child_run(receipt_path):
    """Bounded B1/N1 CUDA diagnostic on the pinned production checkout."""
    import datetime
    import torch
    import importlib

    globals()["torch"] = torch
    root = Path(ROOT).resolve()
    receipt_path = Path(receipt_path)
    receipt = {
        "status": "running",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "commit_expected": EXPECTED_COMMIT,
        "diagnostics": {},
        "public_steps": [],
        "timing_notes": [
            "Direct component timings are isolated and not additive.",
            "Reset, controller, lidar, observation, physics, collision and public-step timings overlap in overall workload, not in measured component boundaries.",
            "Three synchronized repeats per diagnostic stage. Repeat 1 is not "
            "guaranteed cold because prior stages may have warmed kernels, caches, "
            "or allocator state. Memory peaks are process/device allocator observations "
            "and may include retained outputs and audit cache allocations from earlier "
            "stages; they are not isolated per-component memory attribution. "
            "Component timings are standalone measurements, not additive estimates "
            "of public-step cost.",
            "Component input states are fixed snapshots from reset or one standalone physics step.",
            "Controller calculation is excluded from public-step timing.",
        ],
    }
    save_json(receipt_path, receipt)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback prohibited")
        torch.cuda.set_device(torch.device("cuda:0"))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        head = git_guard(root)
        manifest_before = source_manifest(root)
        receipt["commit"] = head
        receipt["source_manifest_before"] = manifest_before
        receipt["runtime"] = {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_index": 0,
            "dtype": "torch.float32",
            "num_worlds": 1,
            "num_agents": 1,
        }
        save_json(receipt_path, receipt)

        benchmark = importlib.import_module("benchmarks.environment_benchmark")
        envmod = importlib.import_module("lawless.environment")
        lidarmod = importlib.import_module("lawless.sensors.lidar")
        obsmod = importlib.import_module("lawless.sensors.observations")
        physicsmod = importlib.import_module("lawless.physics.simulator")
        collisionmod = importlib.import_module("lawless.world.collision")

        modules = {
            "benchmark": benchmark, "environment": envmod,
            "lidar": lidarmod, "observations": obsmod,
            "physics": physicsmod, "collision": collisionmod,
        }
        receipt["module_files"] = {}
        for name, module in modules.items():
            module_file = Path(module.__file__).resolve()
            if not module_file.is_relative_to(root):
                raise RuntimeError(f"{name} module outside checkout: {module_file}")
            receipt["module_files"][name] = str(module_file)

        config = benchmark.create_a21_config(seed=7)
        reset_value, reset_samples = timed_cuda_call(
            torch,
            lambda: envmod.reset(
                config, num_worlds=1, num_agents=1,
                device=torch.device("cuda:0"), dtype=torch.float32,
            ),
            repeats=3,
        )
        environment, observations, info = reset_value
        assert_tensor_contract(observations, (1, 1, 23), torch.float32)
        assert_tensor_contract(environment.worlds[0].episode.physics.position, (1, 2), torch.float32)
        receipt["diagnostics"]["reset"] = {
            "samples": reset_samples,
            "tensor_audit": tensor_audit(reset_value),
        }
        save_json(receipt_path, receipt)

        target_positions = torch.stack(
            [w.episode.physics.position for w in environment.worlds], dim=0
        ).clone()
        assert_tensor_contract(target_positions, (1, 1, 2), torch.float32)
        receipt["target_positions_audit"] = tensor_audit(target_positions)
        save_json(receipt_path, receipt)

        actions, controller_samples = timed_cuda_call(
            torch,
            lambda: benchmark.a21_controller_actions(
                environment, target_positions
            ),
            repeats=3,
        )
        assert_tensor_contract(actions, (1, 1, 2), torch.float32)
        receipt["diagnostics"]["controller"] = {
            "samples": controller_samples,
            "tensor_audit": tensor_audit(actions),
        }
        save_json(receipt_path, receipt)

        episode = environment.worlds[0].episode
        world = episode.generated_world.world
        lidar_input = episode.physics
        lidar_value, lidar_samples = timed_cuda_call(
            torch,
            lambda: lidarmod.sense_lidar(
                lidar_input, world, config.sensors
            ),
            repeats=3,
        )
        assert_tensor_contract(lidar_value, (1, 8), torch.float32)
        receipt["diagnostics"]["lidar"] = {
            "input_state": "reset",
            "samples": lidar_samples,
            "tensor_audit": tensor_audit(lidar_value),
        }
        save_json(receipt_path, receipt)

        observation_input = episode
        observation_value, observation_samples = timed_cuda_call(
            torch,
            lambda: obsmod.assemble_observations(
                observation_input, lidar_value
            ),
            repeats=3,
        )
        assert_tensor_contract(observation_value, (1, 23), torch.float32)
        receipt["diagnostics"]["observation"] = {
            "input_state": "reset with precomputed lidar",
            "samples": observation_samples,
            "tensor_audit": tensor_audit(observation_value),
        }
        save_json(receipt_path, receipt)

        physics_input = episode.physics
        physics_actions = actions[0]
        physics_value, physics_samples = timed_cuda_call(
            torch,
            lambda: physicsmod.step(
                physics_input, physics_actions,
                environment.worlds[0].gravity,
                environment.worlds[0].physics_config,
            ),
            repeats=3,
        )
        assert_tensor_contract(physics_value[0].position, (1, 2), torch.float32)
        receipt["diagnostics"]["physics"] = {
            "input_state": "reset with precomputed actions",
            "samples": physics_samples,
            "tensor_audit": tensor_audit(physics_value),
        }
        save_json(receipt_path, receipt)

        physics_next = physics_value[0]
        collision_input = physics_next
        collision_value, collision_samples = timed_cuda_call(
            torch,
            lambda: collisionmod.resolve_static_obstacles(
                collision_input, world,
                agent_radius_m=episode.generated_world.agent_body_radius_m,
            ),
            repeats=3,
        )
        receipt["diagnostics"]["collision"] = {
            "input_state": "standalone physics output",
            "samples": collision_samples,
            "tensor_audit": tensor_audit(collision_value),
        }
        save_json(receipt_path, receipt)

        # Public stepping starts from the reset state, not the diagnostic
        # component outputs. Controller computation is outside each step timer.
        environment, observations, info = reset_value
        target_positions = target_positions.clone()
        for transition in range(5):
            actions = benchmark.a21_controller_actions(
                environment, target_positions
            )
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = envmod.step(environment, actions)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            environment, observations, rewards, terminated, truncated, info = result
            assert_tensor_contract(observations, (1, 1, 23), torch.float32)
            assert_tensor_contract(rewards, (1, 1), torch.float32)
            assert_tensor_contract(terminated, (1, 1), torch.bool)
            assert_tensor_contract(truncated, (1, 1), torch.bool)
            audit = tensor_audit(result)
            step_counts = benchmark.per_world_steps(environment)
            completed = benchmark.per_world_completed(environment)
            if step_counts != [transition + 1]:
                raise RuntimeError(f"Unexpected step counts: {step_counts}")
            if completed != [False]:
                raise RuntimeError(f"Unexpected completion state: {completed}")

            receipt["public_steps"].append({
                "transition": transition + 1,
                "seconds": elapsed,
                "tensor_audit": audit,
                "step_counts": step_counts,
                "completed": completed,
            })
            save_json(receipt_path, receipt)

        receipt["status"] = "passed"

    except Exception as exc:
        receipt["status"] = "failed"
        receipt["error"] = repr(exc)
        receipt["traceback"] = traceback.format_exc()

    finally:
        try:
            receipt["source_manifest_after"] = source_manifest(root)
            receipt["source_unchanged"] = (
                receipt.get("source_manifest_before")
                == receipt["source_manifest_after"]
            )
            receipt["git_clean_after"] = not bool(
                subprocess.run(
                    ["git", "-C", str(root), "status", "--porcelain",
                     "--untracked-files=all"],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()
            )
            receipt["head_after"] = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            if (receipt["head_after"] != EXPECTED_COMMIT
                    or not receipt["source_unchanged"]
                    or not receipt["git_clean_after"]):
                receipt["status"] = "failed"
                receipt["source_guard_error"] = (
                    "Post-run HEAD/source/cleanliness guard failed"
                )
        except Exception as guard_exc:
            receipt["status"] = "failed"
            receipt["post_run_guard_error"] = repr(guard_exc)
            receipt["post_run_guard_traceback"] = traceback.format_exc()

        receipt["finished_utc"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        save_json(receipt_path, receipt)

    return receipt

def parent_run():
    """Run the reviewed standalone script in a bounded child and archive evidence."""
    script = Path(__file__).resolve()
    root = Path(ROOT).resolve()
    started = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix="lawless_cuda_probe_"))
    receipt_path = work / "child_receipt.json"
    parent_path = work / "parent_receipt.json"
    archive_path = work / "evidence.zip"
    copied_script = work / script.name
    parent = {
        "status": "running",
        "started_utc": datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat(),
        "work_dir": str(work),
        "script_original": str(script),
        "checkout": str(root),
        "expected_commit": EXPECTED_COMMIT,
    }
    stdout_data = b""
    stderr_data = b""
    child_returncode = None

    try:
        if script.is_relative_to(root):
            raise RuntimeError("Probe script must be outside the checkout")
        parent["script_sha256"] = hashlib.sha256(
            script.read_bytes()
        ).hexdigest()
        git_guard(root)
        copied_script.write_bytes(script.read_bytes())
        parent["copied_script_sha256"] = hashlib.sha256(
            copied_script.read_bytes()
        ).hexdigest()
        if parent["script_sha256"] != parent["copied_script_sha256"]:
            raise RuntimeError("Copied script hash mismatch")

        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(root)

        command = [
            sys.executable, str(copied_script),
            "--child", "--receipt", str(receipt_path),
        ]
        parent["command"] = command
        save_json(parent_path, parent)

        remaining = PARENT_BUDGET - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Parent budget exhausted before child launch")
        try:
            completed = subprocess.run(
                command, cwd=str(root), env=env,
                capture_output=True, timeout=min(CHILD_BUDGET, remaining),
            )
            child_returncode = completed.returncode
            stdout_data = completed.stdout
            stderr_data = completed.stderr
        except subprocess.TimeoutExpired as exc:
            stdout_data = exc.stdout or b""
            stderr_data = exc.stderr or b""
            child_returncode = None
            parent["child_timeout"] = True

        parent["child_returncode"] = child_returncode
        parent["child_stdout"] = stdout_data.decode(
            "utf-8", errors="replace"
        )
        parent["child_stderr"] = stderr_data.decode(
            "utf-8", errors="replace"
        )
        parent["child_receipt_exists"] = receipt_path.is_file()
        child_receipt = (
            json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt_path.is_file() else None
        )
        parent["child_status"] = (
            child_receipt.get("status") if child_receipt else None
        )
        parent["child_receipt"] = child_receipt
        parent["parent_post_child_guard"] = {}
        try:
            parent["parent_post_child_guard"] = {
                "head": git_guard(root),
                "source_manifest": source_manifest(root),
                "clean": not bool(subprocess.run(
                    ["git", "-C", str(root), "status", "--porcelain",
                     "--untracked-files=all"],
                    check=True, capture_output=True, text=True,
                ).stdout.strip()),
            }
            parent["parent_post_child_guard"]["verified"] = (
                parent["parent_post_child_guard"]["head"] == EXPECTED_COMMIT
                and parent["parent_post_child_guard"]["clean"] is True
                and parent["parent_post_child_guard"]["source_manifest"]
                == child_receipt.get("source_manifest_after")
                and child_receipt.get("source_unchanged") is True
                and child_receipt.get("git_clean_after") is True
                and child_receipt.get("head_after") == EXPECTED_COMMIT
            )
        except Exception as guard_exc:
            parent["parent_post_child_guard_error"] = repr(guard_exc)
            parent["parent_post_child_guard"]["verified"] = False

        parent["status"] = (
            "passed" if parent["parent_post_child_guard"]["verified"] is True
            and child_returncode == 0
            and child_receipt is not None
            and child_receipt.get("status") == "passed"
            and child_receipt.get("source_unchanged") is True
            and child_receipt.get("git_clean_after") is True
            and child_receipt.get("head_after") == EXPECTED_COMMIT
            else "failed"
        )
    except Exception as exc:
        parent["status"] = "failed"
        parent["error"] = repr(exc)
        parent["traceback"] = traceback.format_exc()
    finally:
        parent["elapsed_seconds"] = time.monotonic() - started
        if parent["elapsed_seconds"] > PARENT_BUDGET:
            parent["parent_budget_exceeded"] = True
            parent["status"] = "failed"
        parent["finished_utc"] = datetime.datetime.now(
            datetime.timezone.utc
        ).isoformat()
        save_json(parent_path, parent)

        (work / "child_stdout.bin").write_bytes(stdout_data)
        (work / "child_stderr.bin").write_bytes(stderr_data)
        archive_files = [
            (copied_script, copied_script.name),
            (parent_path, parent_path.name),
            (work / "child_stdout.bin", "child_stdout.bin"),
            (work / "child_stderr.bin", "child_stderr.bin"),
        ]
        if receipt_path.is_file():
            archive_files.append((receipt_path, receipt_path.name))

        parent["archive_missing_files"] = [
            str(path) for path, _ in archive_files if not path.is_file()
        ]
        parent["archive_included_files"] = [
            str(path) for path, _ in archive_files if path.is_file()
        ]
        save_json(parent_path, parent)

        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path, arcname in archive_files:
                if path.is_file():
                    archive.write(path, arcname)

        archive_bytes = archive_path.read_bytes()
        archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
        encoded = base64.b64encode(archive_bytes).decode("ascii")
        print("PARENT_STATUS:", parent["status"])
        print("ARCHIVE_PATH:", archive_path)
        print("ARCHIVE_BYTES:", len(archive_bytes))
        print("ARCHIVE_SHA256:", archive_sha256)
        print("BASE64_BEGIN")
        print(encoded)
        print("BASE64_END")

    return 0 if parent.get("status") == "passed" else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.child:
        if args.receipt is None:
            parser.error("--child requires --receipt")
        result = child_run(args.receipt)
        return 0 if result.get("status") == "passed" else 1
    return parent_run()


if __name__ == "__main__":
    raise SystemExit(main())