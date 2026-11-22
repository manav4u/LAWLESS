#!/usr/bin/env python3
"""LAWLESS P1.8-A A1 accounted, bounded baseline.

This is a LIMITED A1 baseline. It is not P1.8-A acceptance.

Scope:
  - Canonical production reset/step API.
  - Explicit root seed propagation into EnvironmentConfig.
  - Actual per-world seed recording.
  - Bounded step count and post-reset wall-time budget.
  - Accounting based on worlds that actually advance.
  - Alive-before-transition accounting.
  - Newly completed worlds separated from persistent inventory.
  - Actual Git revision and tracked/untracked source identity.
  - Dynamic tracked Python inventory.
  - Frozen P0 tag Python source identity.
  - Run-specific receipts and atomic partial checkpoints.
  - Accounting-based workload rates, not uninstrumented throughput.

Not implemented by A1:
  - Controlled 2400-transition / 40-second fixture.
  - Component profiling.
  - Uninstrumented step interval baseline.
  - CUDA peak-memory measurements.
  - CPU/T4 acceptance workload repetitions.
  - Compatible resumability.
  - P1.8-A acceptance.

Run from the repository root:
  python -m benchmarks.environment_benchmark --mode smoke
  python -m benchmarks.environment_benchmark --mode pilot

A run's wall-time budget begins after reset, not before reset.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from lawless.environment.reset import reset
from lawless.environment.step import step
from lawless.world.config import EnvironmentConfig


VERSION = "1.2.0-A1"
FROZEN_P0_TAG = "p0-validated"
DECLARED_SOURCE_REVISION = "ac7e0471ae314460d6f72036a33f9a2e3a198bc5"

EVENT_KEYS = (
    "newly_collected",
    "newly_extracted",
    "failure_this_step",
    "energy_death_this_step",
    "collision_contact",
    "unresolved_contact",
    "boundary_escape",
    "shift_applied_this_step",
)


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return json_safe(dataclasses.asdict(value))

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if hasattr(value, "__dict__"):
        return {
            key: json_safe(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return repr(value)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_sum(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.sum().item())

    if isinstance(value, (int, float, bool)):
        return int(value)

    raise TypeError(
        f"Cannot reduce/count {type(value).__name__}"
    )


def tensor_shape(value: Any) -> list[int]:
    return list(value.shape)


def config_json(config: Any) -> Any:
    return json_safe(config)


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# Git and source identity
# ---------------------------------------------------------------------------

def git_command_bytes(
    root: Path,
    *args: str,
) -> subprocess.CompletedProcess[bytes] | None:
    """Execute Git without text decoding or newline conversion."""
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return None


def git_command(
    root: Path,
    *args: str,
) -> subprocess.CompletedProcess[str] | None:
    """Execute Git for text-oriented inventory and status operations."""
    raw = git_command_bytes(root, *args)

    if raw is None:
        return None

    return subprocess.CompletedProcess(
        raw.args,
        raw.returncode,
        stdout=raw.stdout.decode("utf-8", errors="replace"),
        stderr=raw.stderr.decode("utf-8", errors="replace"),
    )


def git_success(root: Path, *args: str) -> str | None:
    result = git_command(root, *args)

    if result is None or result.returncode != 0:
        return None

    return result.stdout.strip()


def tracked_python_inventory(root: Path) -> list[str] | None:
    result = git_command(root, "ls-files", "--cached", "-z")
    if result is None or result.returncode != 0:
        return None

    paths = result.stdout.split("\0")
    return sorted(
        path
        for path in paths
        if path and path.endswith(".py")
    )


def untracked_python_inventory(root: Path) -> list[str] | None:
    result = git_command(
        root,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    )
    if result is None or result.returncode != 0:
        return None

    paths = result.stdout.split("\0")
    return sorted(
        path
        for path in paths
        if path and path.endswith(".py")
    )


def git_dirty_status(root: Path) -> dict[str, Any]:
    """Avoid interpreting non-repository errors as dirty."""
    unstaged = git_command(root, "diff", "--quiet")
    staged = git_command(root, "diff", "--cached", "--quiet")

    if unstaged is None or staged is None:
        return {
            "status": "unknown",
            "dirty": None,
            "error": "Git command could not be executed",
        }

    if unstaged.returncode not in (0, 1):
        return {
            "status": "error",
            "dirty": None,
            "error": unstaged.stderr.strip(),
        }

    if staged.returncode not in (0, 1):
        return {
            "status": "error",
            "dirty": None,
            "error": staged.stderr.strip(),
        }

    dirty = unstaged.returncode == 1 or staged.returncode == 1

    return {
        "status": "verified",
        "dirty": dirty,
        "error": None,
    }


def file_hash_inventory(
    root: Path,
    paths: list[str],
) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    missing = []

    for relative in sorted(set(paths)):
        path = root / relative

        if not path.is_file():
            entries[relative] = {
                "status": "missing",
                "sha256": None,
            }
            missing.append(relative)
            continue

        try:
            content = path.read_bytes()
        except OSError as exc:
            entries[relative] = {
                "status": "error",
                "sha256": None,
                "error": repr(exc),
            }
            missing.append(relative)
            continue

        entries[relative] = {
            "status": "verified",
            "sha256": sha256_bytes(content),
            "size_bytes": len(content),
        }

    return {
        "status": "verified" if not missing else "incomplete",
        "entries": entries,
        "missing_or_error": missing,
    }


def frozen_tag_python_inventory(root: Path) -> dict[str, Any]:
    """Hash Python files from the frozen tag as raw Git blob bytes."""
    tag_commit = git_success(
        root,
        "rev-parse",
        "--verify",
        f"{FROZEN_P0_TAG}^{{commit}}",
    )

    if tag_commit is None:
        return {
            "status": "unknown",
            "tag": FROZEN_P0_TAG,
            "tag_commit": None,
            "error": "Frozen tag commit could not be resolved",
            "entries": {},
        }

    tree = git_command(
        root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        tag_commit,
    )

    if tree is None or tree.returncode != 0:
        return {
            "status": "unknown",
            "tag": FROZEN_P0_TAG,
            "tag_commit": tag_commit,
            "error": (
                tree.stderr.strip()
                if tree is not None
                else "Git unavailable"
            ),
            "entries": {},
        }

    paths = sorted(
        path
        for path in tree.stdout.split("\0")
        if path and path.endswith(".py")
    )

    entries: dict[str, Any] = {}

    for relative in paths:
        blob = git_command_bytes(
            root,
            "show",
            f"{tag_commit}:{relative}",
        )

        if blob is None or blob.returncode != 0:
            entries[relative] = {
                "status": "error",
                "sha256": None,
                "error": (
                    blob.stderr.decode("utf-8", errors="replace").strip()
                    if blob is not None
                    else "Git unavailable"
                ),
            }
            continue

        content = blob.stdout

        entries[relative] = {
            "status": "verified",
            "sha256": sha256_bytes(content),
            "size_bytes": len(content),
        }

    failures = [
        name
        for name, entry in entries.items()
        if entry["status"] != "verified"
    ]

    return {
        "status": "verified" if not failures else "incomplete",
        "tag": FROZEN_P0_TAG,
        "tag_commit": tag_commit,
        "python_file_count": len(entries),
        "entries": entries,
        "missing_or_error": failures,
    }


def source_identity(root: Path) -> dict[str, Any]:
    tracked = tracked_python_inventory(root)
    untracked = untracked_python_inventory(root)
    dirty_status = git_dirty_status(root)
    revision = git_success(root, "rev-parse", "HEAD")
    branch = git_success(root, "branch", "--show-current")

    if tracked is None:
        tracked_status = "unknown"
        tracked = []
    else:
        tracked_status = "verified"

    if untracked is None:
        untracked_status = "unknown"
        untracked = []
    else:
        untracked_status = "verified"

    benchmark_path = "benchmarks/environment_benchmark.py"

    # Hash all tracked Python sources and untracked Python sources, including
    # this benchmark. No guessed source-path list is used.
    paths = set(tracked) | set(untracked) | {benchmark_path}

    current_hashes = file_hash_inventory(root, list(paths))
    frozen_hashes = frozen_tag_python_inventory(root)

    identity_status = "verified"

    if (
        tracked_status != "verified"
        or untracked_status != "verified"
        or dirty_status["status"] != "verified"
        or current_hashes["status"] != "verified"
        or frozen_hashes["status"] != "verified"
        or revision is None
    ):
        identity_status = "incomplete"

    return {
        "status": identity_status,
        "revision": revision,
        "branch": branch,
        "declared_source_revision": DECLARED_SOURCE_REVISION,
        "declared_revision_matches_checkout": (
            revision == DECLARED_SOURCE_REVISION
            if revision is not None
            else None
        ),
        "dirty_status": dirty_status,
        "tracked_inventory_status": tracked_status,
        "tracked_python_file_count": len(tracked),
        "tracked_python_files": tracked,
        "untracked_inventory_status": untracked_status,
        "untracked_python_files": untracked,
        "current_python_hashes": current_hashes,
        "frozen_p0_python_hashes": frozen_hashes,
        "benchmark_included": (
            benchmark_path in current_hashes["entries"]
        ),
        "identity_complete": identity_status == "verified",
    }


# ---------------------------------------------------------------------------
# Environment and workload
# ---------------------------------------------------------------------------

def create_config(seed: int) -> EnvironmentConfig:
    config = EnvironmentConfig()
    config.seed = seed
    config.validate()
    return config


def make_generator(
    seed: int,
    device: torch.device,
) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def create_actions(
    batch: int,
    candidates: int,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.rand(
        (batch, candidates, 2),
        device=device,
        dtype=dtype,
        generator=generator,
    ).mul_(2).sub_(1)


def per_world_completed(environment: Any) -> list[bool]:
    return [bool(world.completed) for world in environment.worlds]


def per_world_steps(environment: Any) -> list[int]:
    return [int(world.step_count) for world in environment.worlds]


def per_world_seeds(environment: Any) -> list[Any]:
    return [json_safe(world.world_seed) for world in environment.worlds]


def per_world_inventory(
    environment: Any,
    attribute: str,
) -> list[int]:
    counts = []

    for world in environment.worlds:
        if attribute == "alive":
            value = world.episode.physics.alive
        elif attribute == "extracted":
            value = world.episode.embodiment.extracted
        elif attribute == "failed":
            value = world.episode.embodiment.failed
        elif attribute == "truncated":
            value = world.truncated
        else:
            raise ValueError(attribute)

        counts.append(tensor_sum(value))

    return counts


def geometry_inventory(environment: Any) -> list[dict[str, Any]]:
    result = []

    for world_episode in environment.worlds:
        generated = world_episode.episode.generated_world
        obstacles = generated.world.obstacles

        result.append(
            {
                "obstacle_count": int(obstacles.count),
                "core_count": int(generated.core_positions.shape[0]),
                "spawn_count": int(
                    generated.spawn_positions.shape[0]
                ),
                "extraction_radius_m": float(
                    generated.extraction_radius_m
                ),
                "agent_body_radius_m": float(
                    generated.agent_body_radius_m
                ),
            }
        )

    return result


# ---------------------------------------------------------------------------
# Accounting
# ---------------------------------------------------------------------------

def pre_step_counts(environment: Any) -> dict[str, Any]:
    """Capture advancement and alive counts before a step.

    Completed worlds are excluded, including completed/truncated worlds
    whose physical alive mask is still true.
    """
    advancing = [
        not bool(world.completed)
        for world in environment.worlds
    ]

    alive_counts = []

    for index, world in enumerate(environment.worlds):
        if advancing[index]:
            alive_counts.append(
                tensor_sum(world.episode.physics.alive)
            )
        else:
            alive_counts.append(0)

    return {
        "advancing": advancing,
        "alive_at_transition_start": alive_counts,
        "worlds_advancing": sum(advancing),
        "alive_agent_transitions_expected": sum(alive_counts),
    }


def event_counts(info: dict[str, Any]) -> dict[str, int]:
    counts = {}

    for key in EVENT_KEYS:
        value = info.get(key)

        if isinstance(value, torch.Tensor):
            counts[key] = tensor_sum(value)

    return counts


def empty_accounting(
    batch: int,
    candidates: int,
) -> dict[str, Any]:
    return {
        "step_calls": 0,
        "world_step_transitions": 0,
        "candidate_slot_transitions": 0,
        "alive_agent_transitions": 0,
        "newly_completed_worlds": 0,
        "completion_events": [],
        "per_world_step_transitions": [0] * batch,
        "per_world_alive_transitions": [0] * batch,
        "per_world_candidate_slot_transitions": [0] * batch,
        "per_world_completion_step": [None] * batch,
        "transition_event_counts": Counter(),
        "candidate_count_per_world": candidates,
    }


def add_step_accounting(
    accounting: dict[str, Any],
    before: dict[str, Any],
    previous_completed: list[bool],
    environment: Any,
    info: dict[str, Any],
) -> None:
    """Update counts from pre-transition activity and transition events."""
    advancing = before["advancing"]
    alive_counts = before["alive_at_transition_start"]

    world_advances = before["worlds_advancing"]
    alive_transitions = before["alive_agent_transitions_expected"]

    accounting["step_calls"] += 1
    accounting["world_step_transitions"] += world_advances
    accounting["candidate_slot_transitions"] += (
        world_advances * environment.num_agents
    )
    accounting["alive_agent_transitions"] += alive_transitions

    for index, active in enumerate(advancing):
        if not active:
            continue

        accounting["per_world_step_transitions"][index] += 1
        accounting["per_world_alive_transitions"][index] += (
            alive_counts[index]
        )
        accounting["per_world_candidate_slot_transitions"][index] += (
            environment.num_agents
        )

    completed_after = per_world_completed(environment)

    newly_completed = [
        index
        for index, (was_completed, is_completed) in enumerate(
            zip(previous_completed, completed_after)
        )
        if not was_completed and is_completed
    ]

    accounting["newly_completed_worlds"] += len(newly_completed)

    for index in newly_completed:
        completion_step = int(environment.worlds[index].step_count)

        accounting["per_world_completion_step"][index] = (
            completion_step
        )

        accounting["completion_events"].append(
            {
                "world_index": index,
                "step_index": completion_step,
            }
        )

    accounting["transition_event_counts"].update(event_counts(info))


def final_inventory(environment: Any) -> dict[str, Any]:
    completed = per_world_completed(environment)

    return {
        "world_completed": completed,
        "worlds_completed": sum(completed),
        "worlds_total": len(completed),
        "world_step_counts": per_world_steps(environment),
        "physical_alive_per_world": per_world_inventory(
            environment, "alive"
        ),
        "extracted_per_world": per_world_inventory(
            environment, "extracted"
        ),
        "failed_per_world": per_world_inventory(
            environment, "failed"
        ),
        "truncated_per_world": per_world_inventory(
            environment, "truncated"
        ),
    }


# ---------------------------------------------------------------------------
# Receipts
# ---------------------------------------------------------------------------

def atomic_write_json(
    path: Path,
    receipt: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe(receipt),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    os.replace(temporary, path)


def runtime_metadata(device: torch.device) -> dict[str, Any]:
    metadata = {
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "python_version": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }

    if device.type == "cuda" and torch.cuda.is_available():
        index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        properties = torch.cuda.get_device_properties(index)

        metadata["cuda"] = {
            "device_index": index,
            "device_name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(index)),
        }

    return metadata


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def execute_run(
    *,
    root: Path,
    mode: str,
    batch: int,
    candidates: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
    max_steps: int,
    post_reset_wall_budget: float,
    output_path: Path,
) -> dict[str, Any]:
    """Execute a bounded run, retaining partial snapshots.

    The wall-time budget begins after reset. Reset time is separately
    measured and does not consume this post-reset budget.
    """
    validate_run_inputs(
        batch=batch,
        candidates=candidates,
        seed=seed,
        device=device,
        max_steps=max_steps,
        post_reset_wall_budget=post_reset_wall_budget,
    )
    config = create_config(seed)
    action_seed = seed + 1234567

    run: dict[str, Any] = {
        "status": "running",
        "started_utc": utc_now(),
        "mode": mode,
        "seed_requested": seed,
        "root_seed_applied": config.seed,
        "action_seed": action_seed,
        "batch": batch,
        "candidates_per_world": candidates,
        "initialized_slots": batch * candidates,
        "max_steps": max_steps,
        "post_reset_wall_budget_seconds": post_reset_wall_budget,
        "device": str(device),
        "dtype": str(dtype),
        "config": config_json(config),
        "source_identity": source_identity(root),
        "accounting": empty_accounting(batch, candidates),
        "step_call_samples_seconds": [],
        "reset_wall_seconds": None,
        "partial_inventory": None,
        "final_inventory": None,
    }

    accounting = run["accounting"]
    environment = None
    run_start = None

    action_generator = make_generator(action_seed, device)

    try:
        sync(device)
        reset_start = time.perf_counter()

        environment, observations, reset_info = reset(
            config,
            batch,
            candidates,
            device=device,
            dtype=dtype,
        )

        sync(device)
        run["reset_wall_seconds"] = (
            time.perf_counter() - reset_start
        )
        run["reset_info"] = json_safe(reset_info)
        run["actual_world_seeds"] = per_world_seeds(environment)
        run["geometry_inventory"] = geometry_inventory(environment)

        previous_completed = per_world_completed(environment)

        sync(device)
        run_start = time.perf_counter()
        deadline = run_start + post_reset_wall_budget

        for step_index in range(max_steps):
            if environment.all_completed:
                run["termination_reason"] = "all_worlds_completed"
                break

            if time.perf_counter() >= deadline:
                run["termination_reason"] = "post_reset_wall_budget"
                break

            before = pre_step_counts(environment)

            actions = create_actions(
                batch,
                candidates,
                action_generator,
                device,
                dtype,
            )

            # This interval is accounted step-call timing, not uninstrumented
            # API throughput. Accounting is performed after the interval.
            sync(device)
            step_start = time.perf_counter()

            (
                environment,
                observations,
                rewards,
                terminated,
                truncated,
                info,
            ) = step(environment, actions)

            sync(device)
            step_elapsed = time.perf_counter() - step_start
            run["step_call_samples_seconds"].append(step_elapsed)

            add_step_accounting(
                accounting,
                before,
                previous_completed,
                environment,
                info,
            )

            previous_completed = per_world_completed(environment)

            run["partial_inventory"] = final_inventory(environment)
            run["partial_event_counts"] = dict(
                accounting["transition_event_counts"]
            )
            run["partial_elapsed_post_reset_wall_seconds"] = (
                time.perf_counter() - run_start
            )

            atomic_write_json(output_path, run)

            if environment.all_completed:
                run["termination_reason"] = "all_worlds_completed"
                break
        else:
            run["termination_reason"] = "max_steps_reached"

        run["elapsed_post_reset_wall_seconds"] = (
            time.perf_counter() - run_start
        )
        run["final_inventory"] = final_inventory(environment)
        run["final_event_counts"] = dict(
            accounting["transition_event_counts"]
        )

        run["status"] = (
            "completed"
            if run["final_inventory"]["worlds_completed"] == batch
            else "bounded_incomplete"
        )

        run["finished_utc"] = utc_now()

    except KeyboardInterrupt:
        run["status"] = "interrupted"
        run["exception"] = "KeyboardInterrupt"
        run["traceback"] = traceback.format_exc()

        if environment is not None:
            run["partial_inventory"] = final_inventory(environment)

    except Exception as exc:
        run["status"] = "failed"
        run["exception"] = repr(exc)
        run["traceback"] = traceback.format_exc()

        if environment is not None:
            run["partial_inventory"] = final_inventory(environment)

    finally:
        if run_start is not None:
            run["elapsed_post_reset_wall_seconds"] = (
                time.perf_counter() - run_start
            )

        run["accounting"] = {
            **accounting,
            "transition_event_counts": dict(
                accounting["transition_event_counts"]
            ),
        }
        run["finished_utc"] = run.get("finished_utc", utc_now())
        atomic_write_json(output_path, run)

    return run


def run_summary(run: dict[str, Any]) -> dict[str, Any]:
    accounting = run.get("accounting", {})
    elapsed = run.get("elapsed_post_reset_wall_seconds")
    samples = run.get("step_call_samples_seconds", [])

    alive_transitions = accounting.get(
        "alive_agent_transitions", 0
    )
    slot_transitions = accounting.get(
        "candidate_slot_transitions", 0
    )
    world_steps = accounting.get(
        "world_step_transitions", 0
    )

    # These are accounted workload rates. They are NOT uninstrumented
    # throughput measurements and do not support vectorization conclusions.
    accounted_rates = {
        "alive_agent_transitions_per_post_reset_wall_second": (
            alive_transitions / elapsed
            if elapsed and elapsed > 0
            else None
        ),
        "candidate_slot_transitions_per_post_reset_wall_second": (
            slot_transitions / elapsed
            if elapsed and elapsed > 0
            else None
        ),
        "world_step_transitions_per_post_reset_wall_second": (
            world_steps / elapsed
            if elapsed and elapsed > 0
            else None
        ),
        "rate_scope": "accounted_bounded_A1",
    }

    sample_statistics = {
        "count": len(samples),
        "mean_seconds": (
            sum(samples) / len(samples)
            if samples
            else None
        ),
        "min_seconds": min(samples) if samples else None,
        "max_seconds": max(samples) if samples else None,
    }

    return {
        "status": run.get("status"),
        "termination_reason": run.get("termination_reason"),
        "reset_wall_seconds": run.get("reset_wall_seconds"),
        "elapsed_post_reset_wall_seconds": elapsed,
        "accounted_step_call_samples": sample_statistics,
        "accounted_rates": accounted_rates,
        "accounting": accounting,
        "final_inventory": run.get("final_inventory"),
    }


# ---------------------------------------------------------------------------
# CLI and output directory protection
# ---------------------------------------------------------------------------

def preset(mode: str) -> dict[str, Any]:
    if mode == "smoke":
        return {
            "batch": 1,
            "candidates": 4,
            "repetitions": 1,
            "max_steps": 2,
            "post_reset_wall_budget": 60.0,
        }

    if mode == "pilot":
        return {
            "batch": 2,
            "candidates": 16,
            "repetitions": 1,
            "max_steps": 600,
            "post_reset_wall_budget": 300.0,
        }

    raise ValueError(mode)


MAX_SUPPORTED_SEED = (1 << 63) - 1


def validate_wall_budget(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            "post-reset wall-time budget must be finite and positive"
        )


def validate_step_bound(value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(
            "maximum step bound must be a positive integer"
        )


def validate_positive_integer(value: Any, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")


def validate_seed_span(seed: int, repetitions: int) -> None:
    """Validate the root seed and inclusive repetition seed range."""
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or seed > MAX_SUPPORTED_SEED
    ):
        raise ValueError(
            f"root seed must be an integer in [0, {MAX_SUPPORTED_SEED}]"
        )

    validate_positive_integer(repetitions, "repetitions")

    last_seed = seed + repetitions - 1

    if last_seed > MAX_SUPPORTED_SEED:
        raise ValueError(
            "root seed plus repetition span exceeds supported seed range"
        )


def validate_device(device: torch.device) -> None:
    """Reject unsupported device types and invalid CUDA targets."""
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"unsupported benchmark device type: {device.type!r}; "
            "supported device types are cpu and cuda"
        )

    if device.type == "cpu":
        if device.index is not None:
            raise ValueError("CPU device must not specify an index")
        return

    if not torch.cuda.is_available():
        raise ValueError("CUDA requested but CUDA is unavailable")

    device_count = torch.cuda.device_count()

    index = (
        device.index
        if device.index is not None
        else torch.cuda.current_device()
    )

    if isinstance(index, bool) or index < 0 or index >= device_count:
        raise ValueError(
            f"CUDA device index {index} is outside available device range"
        )


def validate_run_inputs(
    *,
    batch: int,
    candidates: int,
    seed: int,
    device: torch.device,
    max_steps: int,
    post_reset_wall_budget: float,
) -> None:
    validate_positive_integer(batch, "batch")
    validate_positive_integer(candidates, "candidates")
    validate_seed_span(seed, 1)
    validate_step_bound(max_steps)
    validate_wall_budget(post_reset_wall_budget)
    validate_device(device)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LAWLESS bounded accounted A1 benchmark"
    )

    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot"),
        default="smoke",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument("--batch", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--post-reset-wall-budget", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="benchmark_results_a1",
    )

    args = parser.parse_args()
    defaults = preset(args.mode)

    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    try:
        for name in ("batch", "candidates", "repetitions"):
            validate_positive_integer(
                getattr(args, name),
                name,
            )

        validate_step_bound(args.max_steps)
        validate_wall_budget(args.post_reset_wall_budget)
        validate_seed_span(args.seed, args.repetitions)

        try:
            device = torch.device(args.device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid device specification {args.device!r}: {exc}"
            ) from exc

        validate_device(device)

    except (ValueError, TypeError) as exc:
        parser.error(str(exc))

    args.device_obj = device
    args.dtype_obj = (
        torch.float32
        if args.dtype == "float32"
        else torch.float64
    )

    return args


def main() -> int:
    args = parse_args()
    try:
        validate_seed_span(args.seed, args.repetitions)
        validate_run_inputs(
            batch=args.batch,
            candidates=args.candidates,
            seed=args.seed,
            device=args.device_obj,
            max_steps=args.max_steps,
            post_reset_wall_budget=args.post_reset_wall_budget,
        )
    except (ValueError, TypeError) as exc:
        raise SystemExit(f"Invalid benchmark configuration: {exc}")
    root = Path.cwd()

    # A unique invocation directory prevents one invocation silently
    # overwriting the receipts from another invocation.
    invocation_id = dt.datetime.now(
        dt.timezone.utc
    ).strftime("%Y%m%dT%H%M%S.%fZ")

    output_dir = Path(args.output_dir) / invocation_id

    if output_dir.exists():
        raise RuntimeError(
            f"Refusing to reuse invocation directory: {output_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=False)

    top_receipt: dict[str, Any] = {
        "benchmark": "LAWLESS P1.8-A A1",
        "status": "running",
        "protocol_status": "A1_LIMITED_BASELINE_ONLY",
        "overall_p1_8_a_status": "INCOMPLETE",
        "benchmark_version": VERSION,
        "invocation_id": invocation_id,
        "started_utc": utc_now(),
        "runtime": runtime_metadata(args.device_obj),
        "source_identity": source_identity(root),
        "protocol_scope": [
            "Canonical reset/step API.",
            "Bounded step count.",
            "Post-reset wall-time budget.",
            "Root seed applied to EnvironmentConfig.",
            "Actual world-seed recording.",
            "Advancement-based workload accounting.",
            "Alive-before-transition counting.",
            "New completion events separated from final inventories.",
            "Actual Git source inventory and identity status.",
            "Frozen P0 tag Python identity.",
            "Partial run receipts.",
            "Accounted workload rates only.",
            "A1 limited baseline only.",
            "Overall P1.8-A acceptance incomplete.",
        ],
        "remaining_protocol_gates": [
            "Controlled full 2400-transition/40-second fixture.",
            "Uninstrumented step interval baseline.",
            "Component profiling and diagnostics.",
            "GPU peak-memory measurements.",
            "CPU and Free T4 controlled baseline.",
            "Compatible resumability.",
            "Vectorization equivalence gates.",
        ],
        "workload": {
            "mode": args.mode,
            "batch": args.batch,
            "candidates": args.candidates,
            "repetitions": args.repetitions,
            "max_steps": args.max_steps,
            "post_reset_wall_budget_seconds": (
                args.post_reset_wall_budget
            ),
            "seed": args.seed,
            "action_mode": "private_generator_uniform[-1,1]",
            "dtype": str(args.dtype_obj),
            "device": str(args.device_obj),
        },
        "runs": [],
    }

    top_path = output_dir / "invocation.json"
    atomic_write_json(top_path, top_receipt)

    for repetition in range(args.repetitions):
        seed = args.seed + repetition
        run_path = output_dir / f"run_{repetition:03d}.json"

        run = execute_run(
            root=root,
            mode=args.mode,
            batch=args.batch,
            candidates=args.candidates,
            seed=seed,
            device=args.device_obj,
            dtype=args.dtype_obj,
            max_steps=args.max_steps,
            post_reset_wall_budget=args.post_reset_wall_budget,
            output_path=run_path,
        )

        top_receipt["runs"].append(
            {
                "run_file": run_path.name,
                "seed": seed,
                "summary": run_summary(run),
            }
        )

        statuses = [
            entry["summary"]["status"]
            for entry in top_receipt["runs"]
        ]

        if all(status == "completed" for status in statuses):
            top_receipt["status"] = "completed"
        elif any(
            status in ("failed", "interrupted")
            for status in statuses
        ):
            top_receipt["status"] = "failed_or_interrupted"
        else:
            top_receipt["status"] = "bounded_incomplete"

        atomic_write_json(top_path, top_receipt)

        if run["status"] in ("failed", "interrupted"):
            break

    top_receipt["finished_utc"] = utc_now()
    top_receipt["overall_p1_8_a_status"] = "INCOMPLETE"
    top_receipt["acceptance_note"] = (
        "A1 only. No vectorization conclusions or P1.8-A acceptance."
    )

    atomic_write_json(top_path, top_receipt)

    print(json.dumps(json_safe(top_receipt), indent=2))
    print(f"\nInvocation receipt: {top_path}")

    return (
        0
        if top_receipt["status"] == "completed"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())