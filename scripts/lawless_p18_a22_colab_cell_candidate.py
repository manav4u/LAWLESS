# LAWLESS A2.2 — exact-checkpoint CPU profile + uninstrumented timing
# One profiler episode, then one full timing warmup and five full measurements.
# All receipts and the evidence archive are outside the cloned checkout.

import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

# The one exact source-checkpoint SHA for this run.
SOURCE_CHECKPOINT = "9b9c3b28a41037671b0806d234e1d728082a9b28"
REPOSITORY_URL = "https://github.com/manav4u/LAWLESS.git"
VALIDATION_BRANCH = "validation/environment-benchmarks"

SEED = 7
BATCH = 1
CANDIDATES = 1
PROFILE_STEPS = 2400
TIMING_WARMUPS = 1
TIMING_REPETITIONS = 5

THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

def run_command(args, *, cwd=None):
    result = subprocess.run(
        [str(x) for x in args],
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(map(str, args))}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result.stdout.strip()

def write_strict_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"
    path.write_text(payload, encoding="utf-8")
    return path

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def git_clean_status(repo):
    return run_command(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
    )

def verify_clean_checkout(repo, expected_sha):
    head = run_command(["git", "rev-parse", "HEAD"], cwd=repo)
    require(head == expected_sha, f"Wrong checkout SHA: {head}")
    status = git_clean_status(repo)
    require(status == "", f"Checkout is not clean:\n{status}")
    run_command(["git", "diff", "--exit-code"], cwd=repo)
    run_command(["git", "diff", "--cached", "--exit-code"], cwd=repo)
    return head

def verify_frozen_identity(identity, expected_sha, label):
    require(isinstance(identity, dict), f"{label}: source identity is not a dict")
    require(identity.get("status") == "verified", f"{label}: source status not verified")
    require(identity.get("identity_complete") is True, f"{label}: identity incomplete")
    require(identity.get("benchmark_included") is True, f"{label}: A2.1 benchmark not included")
    require(identity.get("revision") == expected_sha,
            f"{label}: source revision differs from pinned checkpoint: "
            f"{identity.get('revision')!r}")
    preservation = identity.get("frozen_p0_preservation")
    require(isinstance(preservation, dict), f"{label}: frozen P0 evidence missing")
    require(preservation.get("status") == "verified", f"{label}: frozen P0 not verified")
    require(preservation.get("file_count") == 20, f"{label}: expected 20 frozen P0 files")
    require(preservation.get("compared_file_count") == 20,
            f"{label}: not all frozen P0 files were compared")
    require(preservation.get("mismatched_files") == [],
            f"{label}: frozen P0 mismatches found")

def verify_artifact_map_equal(before, after, label):
    require(isinstance(before, dict) and isinstance(after, dict),
            f"{label}: artifact inventories missing")
    require(before == after, f"{label}: artifact inventory changed")
    require(bool(before), f"{label}: empty artifact inventory")

def verify_profile_receipt(receipt, checkpoint):
    require(receipt.get("schema") == "lawless.a22.component_profile.v2",
            "Unexpected component-profile receipt schema")
    require(receipt.get("status") == "valid", "Component profile status is not valid")
    require(receipt.get("accepted_profile") is True, "Component profile was not accepted")
    require(receipt.get("source_unchanged") is True, "Profile source changed")
    require(receipt.get("artifacts_unchanged") is True, "Profile artifacts changed")
    verify_frozen_identity(receipt.get("source_before"), checkpoint, "Profile source before")
    verify_frozen_identity(receipt.get("source_after"), checkpoint, "Profile source after")
    require(receipt["source_before"] == receipt["source_after"],
            "Profile source identity differs before/after")

    progress = receipt.get("progress", {})
    require(progress.get("transitions_attempted") == PROFILE_STEPS,
            "Profile transition attempts are not exactly 2400")
    require(progress.get("public_step_calls_attempted") == PROFILE_STEPS,
            "Profile public-step attempts are not exactly 2400")
    require(progress.get("public_step_calls_returned") == PROFILE_STEPS,
            "Profile public-step returns are not exactly 2400")
    require(progress.get("world_step_counts") == [PROFILE_STEPS],
            "Profile world transition count is not exactly 2400")
    require(progress.get("world_completed") == [True],
            "Profile world did not complete")

    public = receipt.get("public_step", {})
    require(public.get("attempted_calls") == PROFILE_STEPS,
            "Profile public-step timer attempted count is not 2400")
    require(public.get("successful_calls") == PROFILE_STEPS,
            "Profile public-step timer successful count is not 2400")

    components = receipt.get("components", {})
    expected_components = {
        "physics_step",
        "resolve_static_obstacles",
        "sense_lidar",
        "assemble_observations",
    }
    require(set(components) == expected_components,
            "Profile component timer set differs from the required four hooks")
    for name, timer in components.items():
        require(timer.get("attempted_calls") == PROFILE_STEPS,
                f"{name}: attempted calls are not 2400")
        require(timer.get("successful_calls") == PROFILE_STEPS,
                f"{name}: successful calls are not 2400")
        require(timer.get("timing_valid") is True,
                f"{name}: timer validity failed")
        require(timer.get("error") is None, f"{name}: timer recorded an error")

    restoration = receipt.get("restoration", {})
    require(restoration.get("status") == "verified",
            "Profiler did not verify canonical-hook restoration")
    identities = restoration.get("identities", {})
    expected_identity_names = {
        "physics_step",
        "resolve_static_obstacles",
        "sense_lidar",
        "assemble_observations",
        "module_step",
        "a21_step",
    }
    require(set(identities) == expected_identity_names,
            "Restoration receipt does not cover all six canonical identities")
    require(all(value is True for value in identities.values()),
            "One or more canonical callable identities were not restored")
    require(not restoration.get("errors"), "Restoration recorded errors")
    require(receipt.get("validation", {}).get("restoration") == "verified",
            "Restoration validation gate did not pass")
    verify_artifact_map_equal(
        receipt.get("artifacts_before"),
        receipt.get("artifacts_after"),
        "Profile artifact hashes",
    )

def verify_timing_receipt(receipt, checkpoint):
    require(receipt.get("schema") == "lawless.a22.cpu_timing_receipt",
            "Unexpected timing receipt schema")
    require(receipt.get("status") == "valid", "Timing runner status is not valid")
    require(receipt.get("valid") is True, "Timing runner did not accept the run")

    workload = receipt.get("workload", {})
    expected_workload = {
        "seed": SEED,
        "batch": BATCH,
        "candidates": CANDIDATES,
        "device": "cpu",
        "dtype": "torch.float32",
        "episode_steps": PROFILE_STEPS,
        "warmups": TIMING_WARMUPS,
        "measured_repetitions": TIMING_REPETITIONS,
    }
    for key, expected in expected_workload.items():
        require(workload.get(key) == expected,
                f"Timing workload {key}={workload.get(key)!r}; expected {expected!r}")

    warmups = receipt.get("warmup_repetitions", [])
    measured = receipt.get("measured_repetitions", [])
    require(len(warmups) == TIMING_WARMUPS, "Expected exactly one full warmup")
    require(len(measured) == TIMING_REPETITIONS,
            "Expected exactly five full measured episodes")

    all_repetitions = warmups + measured
    fingerprints = set()
    for index, item in enumerate(all_repetitions):
        phase = "warmup" if index < TIMING_WARMUPS else "measured"
        require(item.get("phase") == phase, f"Repetition {index}: wrong phase")
        require(item.get("valid") is True and item.get("status") == "valid",
                f"Repetition {index}: invalid episode: {item.get('error')}")
        require(item.get("seed") == SEED, f"Repetition {index}: wrong seed")
        require(item.get("batch") == BATCH, f"Repetition {index}: wrong batch")
        require(item.get("candidates") == CANDIDATES,
                f"Repetition {index}: wrong candidate count")
        require(item.get("transitions_attempted") == PROFILE_STEPS,
                f"Repetition {index}: not exactly 2400 transitions attempted")
        require(item.get("public_step_calls_returned") == PROFILE_STEPS,
                f"Repetition {index}: not exactly 2400 public calls returned")
        require(item.get("episode_seconds") is not None
                and float(item["episode_seconds"]) > 0,
                f"Repetition {index}: no accepted positive episode duration")

        before = item.get("source_identity_before")
        after = item.get("source_identity_after")
        verify_frozen_identity(before, checkpoint, f"Timing repetition {index} before")
        verify_frozen_identity(after, checkpoint, f"Timing repetition {index} after")
        require(item.get("source_fingerprint_before")
                == item.get("source_fingerprint_after"),
                f"Repetition {index}: source identity changed during episode")
        fingerprints.add(item.get("source_fingerprint_before"))

    require(len(fingerprints) == 1,
            "Warmup and measured episodes do not share one source fingerprint")
    require(receipt.get("common_source_fingerprint") in fingerprints,
            "Timing receipt common source fingerprint is inconsistent")

    summary = receipt.get("summary", {})
    require(summary.get("measured_repetition_count") == TIMING_REPETITIONS,
            "Timing summary does not contain five measured repetitions")
    require(summary.get("valid_repetition_count") == TIMING_REPETITIONS,
            "Timing summary does not validate all five measured repetitions")
    require(summary.get("invalid_repetition_count") == 0,
            "Timing summary contains invalid measured repetitions")
    require(summary.get("all_measured_repetitions_valid") is True,
            "Timing summary did not accept all measured repetitions")
    for key in (
        "median_episode_seconds",
        "median_episode_steps_per_second",
        "median_world_steps_per_second",
    ):
        require(summary.get(key) is not None and float(summary[key]) > 0,
                f"Timing summary has no valid {key}")

    verify_artifact_map_equal(
        receipt.get("artifact_hashes_before"),
        receipt.get("artifact_hashes_after"),
        "Timing artifact hashes",
    )

    scope = receipt.get("scope", {})
    require(scope.get("production_code_modified") is False,
            "Timing scope reports production code modification")
    require(scope.get("a21_code_modified") is False,
            "Timing scope reports A2.1 modification")
    require(scope.get("frozen_p0_modified") is False,
            "Timing scope reports frozen P0 modification")
    require(scope.get("component_profiling_included") is False,
            "Timing run was not uninstrumented")

# Keep all evidence in a sibling directory, never inside the checkout.
workspace = Path(tempfile.mkdtemp(prefix="lawless_a22_colab_", dir="/content"))
repo = workspace / "checkout"
evidence = workspace / "evidence"
evidence.mkdir(parents=True)
profile_path = evidence / "a22_cpu_component_profile.json"
timing_path = evidence / "a22_cpu_uninstrumented_timing.json"
manifest_path = evidence / "run_manifest.json"
failure_path = evidence / "failure.json"
zip_path = workspace / "LAWLESS_A22_CPU_evidence.zip"

failure = None
profile_receipt = None
timing_receipt = None
timing_module = None
profiler_module = None
step_module = None
canonical_before = None
runtime_before = None
runtime_after = None
git_head = None
remote_url = None
initial_tree_status = None
final_tree_status = None

try:
    # Clone only the public validation branch, then detach at the exact commit.
    run_command([
        "git", "clone", "--single-branch",
        "--branch", VALIDATION_BRANCH,
        REPOSITORY_URL, repo,
    ])
    remote_url = run_command(["git", "remote", "get-url", "origin"], cwd=repo)
    require(remote_url.rstrip("/").removesuffix(".git").lower()
            == REPOSITORY_URL.rstrip("/").removesuffix(".git").lower(),
            f"Unexpected origin URL: {remote_url}")

    branch_ref = f"origin/{VALIDATION_BRANCH}"
    run_command(["git", "fetch", "origin", VALIDATION_BRANCH], cwd=repo)
    run_command(
        ["git", "merge-base", "--is-ancestor", SOURCE_CHECKPOINT, branch_ref],
        cwd=repo,
    )
    run_command(["git", "checkout", "--detach", SOURCE_CHECKPOINT], cwd=repo)
    git_head = verify_clean_checkout(repo, SOURCE_CHECKPOINT)
    initial_tree_status = git_clean_status(repo)

    # Imports are from this exact clean checkout. No pip installs or upgrades.
    sys.path.insert(0, str(repo))
    importlib.invalidate_caches()

    from benchmarks import environment_timing_a22 as timing_module
    from benchmarks import environment_profile_a22 as profiler_module

    require(Path(timing_module.__file__).resolve()
            == (repo / "benchmarks/environment_timing_a22.py").resolve(),
            "Timing module was not imported from the pinned checkout")
    require(Path(profiler_module.__file__).resolve()
            == (repo / "benchmarks/environment_profile_a22.py").resolve(),
            "Profiler module was not imported from the pinned checkout")

    # Capture actual runtime/thread configuration; do not change it.
    runtime_before = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_version": str(torch.__version__),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV_NAMES},
        "timing_runner_runtime": timing_module._runtime_metadata(),
    }

    # Validate source identity/frozen P0 before starting either workload.
    initial_identity = timing_module._get_source_identity(repo)
    verify_frozen_identity(initial_identity, SOURCE_CHECKPOINT, "Initial checkout")

    # Snapshot all six canonical callable objects before profiling.
    step_module = profiler_module._load_step_module()
    require(not profiler_module._ACTIVE_MODULES,
            "Profiler active-module guard was not clear before profiling")
    require(not profiler_module._CONTAMINATED_MODULES,
            "Profiler contamination guard was not clear before profiling")
    canonical_before = profiler_module._original_callables(step_module)

    # One real full-length component profile. Receipt path is outside checkout.
    profile_receipt = profiler_module.profile_episode(
        seed=SEED,
        batch=BATCH,
        candidates=CANDIDATES,
        output_path=profile_path,
    )
    require(profile_path.is_file(), "Profiler did not persist its raw receipt")
    verify_profile_receipt(profile_receipt, SOURCE_CHECKPOINT)

    # Verify exact identities against the pre-profile objects and current owners.
    for name in profiler_module.COMPONENT_NAMES:
        require(getattr(step_module, name) is canonical_before[name],
                f"Canonical hook {name} was not restored by identity")
    require(step_module.step is canonical_before["module_step"],
            "Production module step callable was not restored by identity")
    require(timing_module.a21.step is canonical_before["a21_step"],
            "A2.1 public step callable was not restored by identity")

    canonical_after_profile = profiler_module._original_callables(step_module)
    for name, original in canonical_before.items():
        require(canonical_after_profile[name] is original,
                f"Canonical callable {name} changed after profiling")
    require(not profiler_module._ACTIVE_MODULES,
            "Profiler active-module guard remains set after profiling")
    require(not profiler_module._CONTAMINATED_MODULES,
            "Profiler contamination guard remains set after profiling")

    # Recheck pinned source/tree immediately before the uninstrumented timing run.
    verify_clean_checkout(repo, SOURCE_CHECKPOINT)
    pre_timing_identity = timing_module._get_source_identity(repo)
    verify_frozen_identity(pre_timing_identity, SOURCE_CHECKPOINT,
                           "Pre-timing source")

    # SAME PROCESS: one complete warmup, then exactly five complete measured runs.
    timing_receipt = timing_module.run_benchmark(
        root=repo,
        seed=SEED,
        batch=BATCH,
        candidates=CANDIDATES,
        warmups=TIMING_WARMUPS,
        repetitions=TIMING_REPETITIONS,
    )
    timing_module.write_receipt(timing_receipt, timing_path)
    require(timing_path.is_file(), "Timing runner did not persist its raw receipt")
    verify_timing_receipt(timing_receipt, SOURCE_CHECKPOINT)

    # Timing must not have left wrappers installed or contaminated the process.
    for name in profiler_module.COMPONENT_NAMES:
        require(getattr(step_module, name) is canonical_before[name],
                f"Canonical hook {name} changed during timing")
    require(step_module.step is canonical_before["module_step"],
            "Production step identity changed during timing")
    require(timing_module.a21.step is canonical_before["a21_step"],
            "A2.1 step identity changed during timing")
    require(not profiler_module._ACTIVE_MODULES,
            "Profiler active-module guard is not clear after timing")
    require(not profiler_module._CONTAMINATED_MODULES,
            "Profiler contamination guard is not clear after timing")

    final_identity = timing_module._get_source_identity(repo)
    verify_frozen_identity(final_identity, SOURCE_CHECKPOINT, "Final source")
    require(timing_module._identity_fingerprint(initial_identity)
            == timing_module._identity_fingerprint(final_identity),
            "Source identity changed across the complete validation run")

    final_tree_status = verify_clean_checkout(repo, SOURCE_CHECKPOINT)
    runtime_after = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV_NAMES},
        "timing_runner_runtime": timing_module._runtime_metadata(),
    }
    require(runtime_after["torch_num_threads"] == runtime_before["torch_num_threads"],
            "Torch intra-op thread count changed during validation")
    require(runtime_after["torch_num_interop_threads"]
            == runtime_before["torch_num_interop_threads"],
            "Torch inter-op thread count changed during validation")
    require(runtime_after["thread_environment"] == runtime_before["thread_environment"],
            "Thread environment variables changed during validation")

except BaseException as exc:
    failure = {
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
        "captured_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_strict_json(failure_path, failure)

finally:
    # Preserve any returned raw receipts, including invalid receipts.
    if profile_receipt is not None:
        write_strict_json(profile_path, profile_receipt)
    if timing_receipt is not None and timing_module is not None:
        timing_module.write_receipt(timing_receipt, timing_path)

    if repo.exists():
        try:
            final_tree_status = git_clean_status(repo)
        except BaseException as exc:
            final_tree_status = f"STATUS_CHECK_FAILED: {type(exc).__name__}: {exc}"

    manifest = {
        "schema": "lawless.a22.colab_cpu_evidence_manifest.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "valid" if failure is None else "invalid",
        "source_checkpoint": SOURCE_CHECKPOINT,
        "repository_url": REPOSITORY_URL,
        "validation_branch": VALIDATION_BRANCH,
        "remote_url": remote_url,
        "git_head": git_head,
        "initial_tree_status": initial_tree_status,
        "final_tree_status": final_tree_status,
        "workload": {
            "seed": SEED,
            "batch": BATCH,
            "candidates": CANDIDATES,
            "device": "cpu",
            "dtype": "torch.float32",
            "episode_steps": PROFILE_STEPS,
            "component_profile_episodes": 1,
            "timing_full_warmup_episodes": TIMING_WARMUPS,
            "timing_measured_full_episodes": TIMING_REPETITIONS,
        },
        "runtime_before": runtime_before,
        "runtime_after": runtime_after,
        "profile_receipt_file": profile_path.name if profile_path.exists() else None,
        "profile_receipt_sha256": sha256_file(profile_path) if profile_path.exists() else None,
        "timing_receipt_file": timing_path.name if timing_path.exists() else None,
        "timing_receipt_sha256": sha256_file(timing_path) if timing_path.exists() else None,
        "failure": failure,
        "scope_limits": [
            "No production, A2.1, or frozen P0 source edits",
            "Component profile is instrumented and is not an uninstrumented baseline",
            "Component intervals are inclusive and nested; do not sum them",
            "Timing consists of one full warmup and five full measured episodes",
            "No throughput extrapolation or scaling claim",
            "No main merge, P1.8-A acceptance, or P6 release claim",
        ],
    }
    write_strict_json(manifest_path, manifest)

# Package all available evidence, including failure details if a gate failed.
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(evidence.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=path.relative_to(evidence).as_posix())

print("Evidence archive:", zip_path)
print("Evidence files:")
for path in sorted(evidence.iterdir()):
    if path.is_file():
        print(f"  {path.name}  ({path.stat().st_size:,} bytes)")

# Trigger the Colab download before raising any validation failure.
try:
    from google.colab import files
    files.download(str(zip_path))
except Exception as download_error:
    print("Automatic Colab download was unavailable:", repr(download_error))
    print("The archive remains at:", zip_path)

if failure is not None:
    raise RuntimeError(
        "A2.2 CPU evidence run FAILED closed. Inspect failure.json and the "
        f"raw receipts in {evidence}. No acceptance claim is made."
    ) from None

print("\nA2.2 CPU evidence run: VALID")
print("Pinned checkpoint:", SOURCE_CHECKPOINT)
print("Profile: 1 full 2400-step B1/N1 seed-7 CPU episode")
print("Timing: 1 full warmup + 5 full measured 2400-step B1/N1 seed-7 CPU episodes")
print("Profile restoration: six canonical callable identities verified; guards clear")
print("Checkout: clean; frozen P0 and source/artifact identity gates passed")
print("No merge, P1.8-A acceptance, P6 release, or throughput extrapolation is claimed.")
