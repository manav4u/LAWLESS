#!/usr/bin/env python3
"""Finite benchmark and attempt-count evidence for LAWLESS full layouts.

Run from the repository root:

    python scripts/benchmark_full_layout.py
    python scripts/benchmark_full_layout.py --device cuda --repeats 2
    python scripts/benchmark_full_layout.py --obstacles 1 --spawns 0 \
        --cores 0 --seeds 7 --dtypes float32 --repeats 1

Ordinary timings use direct, uninstrumented generate_world calls.

Attempt counts are collected in separate instrumented calls. Instrumentation
does not consume random values. It counts distinct obstacle candidate-loop
indices observed at torch.rand calls inside the obstacle helper, and counts
calls to full_generator._candidate_center for the other categories.

The script compares exact output tensors and the private CPU generator's
final state between instrumented and uninstrumented executions.

Counts are reported per category and per placement index. Obstacle totals
include all attempted obstacles, including earlier successful placements
when a later obstacle exhausts its budget.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import json
import statistics
import time
from collections import defaultdict
from typing import Any

import torch

import lawless.world.full_generator as full_generator
import lawless.world.generator as obstacle_generator
from lawless.world.config import (
    EmbodimentConfig,
    EnvironmentConfig,
    ObjectiveConfig,
    WorldConfig,
)


CATEGORIES = ("obstacle", "spawn", "extraction", "core")


class _TorchProxy:
    """Delegate torch attributes while capturing Generator instances."""

    def __init__(self) -> None:
        self.generators: list[torch.Generator] = []

    def __getattr__(self, name: str) -> Any:
        if name == "Generator":
            return self._make_generator
        return getattr(torch, name)

    def _make_generator(self, *args: Any, **kwargs: Any) -> torch.Generator:
        generator = torch.Generator(*args, **kwargs)
        self.generators.append(generator)
        return generator


class _ObstacleTorchProxy:
    """Delegate torch while counting distinct obstacle candidate attempts."""

    def __init__(
        self,
        *,
        helper_code: Any,
        outer_index_name: str,
        attempt_index_name: str,
        enabled: bool,
    ) -> None:
        self.helper_code = helper_code
        self.outer_index_name = outer_index_name
        self.attempt_index_name = attempt_index_name
        self.enabled = enabled
        self.original_rand = torch.rand
        self.seen_attempts: set[tuple[int, int]] = set()

    def __getattr__(self, name: str) -> Any:
        if name == "rand":
            return self._rand
        return getattr(torch, name)

    def _rand(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.enabled:
            frame = inspect.currentframe().f_back
            if frame is not None and frame.f_code is self.helper_code:
                local_vars = frame.f_locals
                if (
                    self.outer_index_name in local_vars
                    and self.attempt_index_name in local_vars
                ):
                    obstacle_index = int(
                        local_vars[self.outer_index_name]
                    )
                    attempt_index = int(
                        local_vars[self.attempt_index_name]
                    )
                    self.seen_attempts.add(
                        (obstacle_index, attempt_index)
                    )

        return self.original_rand(*args, **kwargs)


def _obstacle_loop_variables() -> tuple[Any, str, str]:
    """Find the obstacle helper's nested obstacle/candidate loop variables.

    Counting is based on distinct loop-variable values observed at random
    draws, not Python line-event multiplicity.
    """
    helper = obstacle_generator._generate_obstacles_with_rng
    source = inspect.cleandoc(inspect.getsource(helper))
    tree = ast.parse(source)

    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    candidate_loops: list[ast.For] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue

        if "max_attempts_per_obstacle" not in ast.unparse(node.iter):
            continue

        candidate_loops.append(node)

    if len(candidate_loops) != 1:
        raise RuntimeError(
            "Could not uniquely identify the obstacle candidate loop. "
            "Attempt counts are unavailable; refusing to report estimates."
        )

    candidate_loop = candidate_loops[0]
    outer_loop = parents.get(id(candidate_loop))

    while outer_loop is not None and not isinstance(outer_loop, ast.For):
        outer_loop = parents.get(id(outer_loop))

    if not isinstance(outer_loop, ast.For):
        raise RuntimeError(
            "Could not identify the obstacle-index loop enclosing the "
            "candidate loop."
        )

    if not isinstance(candidate_loop.target, ast.Name):
        raise RuntimeError(
            "Obstacle candidate-loop target must be a simple variable."
        )

    if not isinstance(outer_loop.target, ast.Name):
        raise RuntimeError(
            "Obstacle outer-loop target must be a simple variable."
        )

    return (
        helper.__code__,
        outer_loop.target.id,
        candidate_loop.target.id,
    )


def _make_config(
    *,
    obstacles: int,
    cores: int,
    seed: int,
) -> EnvironmentConfig:
    return EnvironmentConfig(
        seed=seed,
        world=WorldConfig(
            width_m=100.0,
            height_m=100.0,
            obstacle_count=obstacles,
            obstacle_radius_min_m=1.0,
            obstacle_radius_max_m=3.0,
        ),
        embodiment=EmbodimentConfig(agent_radius_m=0.5),
        objectives=ObjectiveConfig(
            core_count=cores,
            core_radius_m=0.75,
            extraction_radius_m=3.0,
        ),
    )


def _same_layout(left: Any, right: Any) -> bool:
    """Compare every returned layout tensor and scalar exactly."""
    if type(left) is not type(right):
        return False

    if left.seed != right.seed:
        return False

    if left.agent_body_radius_m != right.agent_body_radius_m:
        return False

    if left.extraction_radius_m != right.extraction_radius_m:
        return False

    if (
        left.world.width_m != right.world.width_m
        or left.world.height_m != right.world.height_m
    ):
        return False

    tensor_pairs = (
        (
            left.world.obstacles.centers,
            right.world.obstacles.centers,
        ),
        (
            left.world.obstacles.radii,
            right.world.obstacles.radii,
        ),
        (left.spawn_positions, right.spawn_positions),
        (left.extraction_center, right.extraction_center),
        (left.core_positions, right.core_positions),
        (left.core_radii, right.core_radii),
    )

    return all(
        a.dtype == b.dtype
        and a.device == b.device
        and torch.equal(a, b)
        for a, b in tensor_pairs
    )


def _format_attempt_counts(
    *,
    obstacle_seen: set[tuple[int, int]],
    placement_counts: dict[str, dict[int, int]],
    obstacle_count: int,
    generator_created: bool,
    obstacle_generation_failed: bool = False,
) -> dict[str, dict[str, Any]]:
    """Format attempts, allowing only a verified prefix on obstacle failure."""
    obstacle_by_index: dict[int, int] = defaultdict(int)
    attempts_by_obstacle: dict[int, set[int]] = defaultdict(set)

    for obstacle_index, attempt_index in obstacle_seen:
        if obstacle_index < 0 or attempt_index < 0:
            raise RuntimeError(
                "Observed a negative obstacle or candidate-loop index; "
                "refusing to report unverifiable attempt counts."
            )

        obstacle_by_index[obstacle_index] += 1
        attempts_by_obstacle[obstacle_index].add(attempt_index)

    if not generator_created:
        # Validation failed before the private RNG was created.
        if obstacle_seen:
            raise RuntimeError(
                "Obstacle attempts were observed although no private "
                "generator was created; instrumentation state is "
                "inconsistent."
            )
        obstacle_by_index = defaultdict(int)

    elif obstacle_count == 0:
        if obstacle_seen:
            raise RuntimeError(
                "Observed obstacle attempts when obstacle_count is zero; "
                "instrumentation state is inconsistent."
            )

    else:
        if not obstacle_seen:
            raise RuntimeError(
                "Generation created its private RNG, but no obstacle "
                "candidate attempts were observed. The obstacle helper may "
                "no longer draw with torch.rand; refusing to report a "
                "misleading obstacle count."
            )

        observed_indices = set(obstacle_by_index)
        expected_indices = set(range(obstacle_count))

        # A failed obstacle placement may legitimately stop generation
        # before later obstacle indices are visited. No other incomplete
        # obstacle sequence is accepted.
        if obstacle_generation_failed:
            expected_prefix = set(range(len(observed_indices)))
            if observed_indices != expected_prefix:
                raise RuntimeError(
                    "Observed obstacle indices are not a contiguous prefix "
                    "starting at zero; refusing to report unverifiable "
                    "attempt counts."
                )
        elif observed_indices != expected_indices:
            raise RuntimeError(
                "Successful or non-obstacle-failed generation did not "
                "observe every requested obstacle index; refusing to "
                "report incomplete attempt counts."
            )

        # Candidate attempts for each observed obstacle must also form a
        # contiguous zero-based sequence. This detects missing rand hooks
        # within an otherwise observed obstacle index.
        for obstacle_index, attempt_indices in attempts_by_obstacle.items():
            expected_attempts = set(range(len(attempt_indices)))
            if attempt_indices != expected_attempts:
                raise RuntimeError(
                    f"Observed candidate attempts for obstacle "
                    f"{obstacle_index} are not contiguous from zero; "
                    "refusing to report unverifiable attempt counts."
                )

    result: dict[str, dict[str, Any]] = {}

    raw: dict[str, dict[int, int]] = {
        "obstacle": dict(obstacle_by_index),
        **placement_counts,
    }

    for category in CATEGORIES:
        by_index = raw.get(category, {})
        result[category] = {
            "total": sum(by_index.values()),
            "by_index": {
                str(index): by_index[index]
                for index in sorted(by_index)
            },
        }

    return result


def _run_with_state_capture(
    *,
    config: EnvironmentConfig,
    seed: int,
    spawn_count: int,
    dtype: torch.dtype,
    device: str,
    budgets: dict[str, int],
    instrument: bool,
) -> dict[str, Any]:
    """Run one call and capture output, attempts, errors and RNG state."""
    proxy = _TorchProxy()
    helper_code, outer_index_name, attempt_index_name = (
        _obstacle_loop_variables()
    )

    obstacle_proxy = _ObstacleTorchProxy(
        helper_code=helper_code,
        outer_index_name=outer_index_name,
        attempt_index_name=attempt_index_name,
        enabled=instrument,
    )

    placement_counts: dict[str, dict[int, int]] = {
        "spawn": defaultdict(int),
        "extraction": defaultdict(int),
        "core": defaultdict(int),
    }

    original_full_torch = full_generator.torch
    original_obstacle_torch = obstacle_generator.torch
    original_candidate = full_generator._candidate_center

    def counted_candidate_center(**kwargs: Any) -> torch.Tensor:
        caller = inspect.currentframe().f_back
        category = caller.f_locals.get("category") if caller else None

        if instrument and category in placement_counts:
            index = caller.f_locals.get("index", 0)
            placement_counts[category][int(index)] += 1

        return original_candidate(**kwargs)

    result = None
    error_type = None
    error_message = None
    generator_state = None
    attempt_counts = None

    try:
        full_generator.torch = proxy
        obstacle_generator.torch = obstacle_proxy

        if instrument:
            full_generator._candidate_center = counted_candidate_center

        try:
            result = full_generator.generate_world(
                config,
                seed=seed,
                spawn_count=spawn_count,
                device=device,
                dtype=dtype,
                max_attempts_per_obstacle=budgets["obstacle"],
                max_attempts_per_spawn=budgets["spawn"],
                max_attempts_per_extraction=budgets["extraction"],
                max_attempts_per_core=budgets["core"],
            )
        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)
        finally:
            if proxy.generators:
                generator_state = proxy.generators[-1].get_state().clone()

        if instrument:
            obstacle_generation_failed = (
                error_type == "WorldGenerationError"
                and "category='obstacle'" in (error_message or "")
            )

            attempt_counts = _format_attempt_counts(
                obstacle_seen=obstacle_proxy.seen_attempts,
                placement_counts=placement_counts,
                obstacle_count=config.world.obstacle_count,
                generator_created=bool(proxy.generators),
                obstacle_generation_failed=obstacle_generation_failed,
            )

    finally:
        full_generator.torch = original_full_torch
        obstacle_generator.torch = original_obstacle_torch
        full_generator._candidate_center = original_candidate

    return {
        "result": result,
        "error_type": error_type,
        "error_message": error_message,
        "generator_state": generator_state,
        "attempt_counts": attempt_counts,
    }


def _same_outcome(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["error_type"] != right["error_type"]:
        return False

    if left["error_message"] != right["error_message"]:
        return False

    if left["error_type"] is None:
        if not _same_layout(left["result"], right["result"]):
            return False

    left_state = left["generator_state"]
    right_state = right["generator_state"]

    if left_state is None or right_state is None:
        return left_state is right_state

    return torch.equal(left_state, right_state)


def _ordinary_timings(
    *,
    config: EnvironmentConfig,
    seed: int,
    spawn_count: int,
    dtype: torch.dtype,
    device: str,
    budgets: dict[str, int],
    repeats: int,
) -> dict[str, Any]:
    """Measure only ordinary, uninstrumented generate_world calls."""
    samples: list[float] = []
    failures: list[dict[str, Any]] = []

    for repeat in range(repeats):
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)

        started = time.perf_counter()
        error = None

        try:
            full_generator.generate_world(
                config,
                seed=seed,
                spawn_count=spawn_count,
                device=device,
                dtype=dtype,
                max_attempts_per_obstacle=budgets["obstacle"],
                max_attempts_per_spawn=budgets["spawn"],
                max_attempts_per_extraction=budgets["extraction"],
                max_attempts_per_core=budgets["core"],
            )
        except Exception as exc:
            error = {
                "repeat": repeat,
                "type": type(exc).__name__,
                "message": str(exc),
            }
        finally:
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)

            elapsed = time.perf_counter() - started
            samples.append(elapsed)

            if error is not None:
                failures.append(error)

    return {
        "samples_seconds": samples,
        "min_seconds": min(samples) if samples else None,
        "median_seconds": statistics.median(samples) if samples else None,
        "mean_seconds": statistics.mean(samples) if samples else None,
        "failures": failures,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LAWLESS full-layout generation."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42])
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=("float32", "float64"),
        default=["float32", "float64"],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--obstacles", type=int, default=20)
    parser.add_argument("--spawns", type=int, default=4)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=10_000)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")

    if min(args.obstacles, args.spawns, args.cores) < 0:
        raise SystemExit("counts must be nonnegative")

    if args.max_attempts <= 0:
        raise SystemExit("--max-attempts must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but is not available")

    if args.device == "cuda":
        torch.cuda.init()

    budgets = {
        "obstacle": args.max_attempts,
        "spawn": args.max_attempts,
        "extraction": args.max_attempts,
        "core": args.max_attempts,
    }

    print(
        json.dumps(
            {
                "record": "benchmark_metadata",
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "device": args.device,
                "cuda_device_name": (
                    torch.cuda.get_device_name()
                    if args.device == "cuda"
                    else None
                ),
                "repeats": args.repeats,
                "seeds": args.seeds,
                "dtypes": args.dtypes,
                "counts": {
                    "obstacles": args.obstacles,
                    "spawns": args.spawns,
                    "cores": args.cores,
                },
                "candidate_budgets": budgets,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for seed in args.seeds:
        config = _make_config(
            obstacles=args.obstacles,
            cores=args.cores,
            seed=seed,
        )

        for dtype_name in args.dtypes:
            dtype = (
                torch.float32
                if dtype_name == "float32"
                else torch.float64
            )

            # This section is deliberately uninstrumented.
            timing = _ordinary_timings(
                config=config,
                seed=seed,
                spawn_count=args.spawns,
                dtype=dtype,
                device=args.device,
                budgets=budgets,
                repeats=args.repeats,
            )

            # Separate correctness/evidence calls; never timed.
            untraced = _run_with_state_capture(
                config=config,
                seed=seed,
                spawn_count=args.spawns,
                dtype=dtype,
                device=args.device,
                budgets=budgets,
                instrument=False,
            )
            traced = _run_with_state_capture(
                config=config,
                seed=seed,
                spawn_count=args.spawns,
                dtype=dtype,
                device=args.device,
                budgets=budgets,
                instrument=True,
            )

            record = {
                "record": "full_layout_case",
                "seed": seed,
                "dtype": dtype_name,
                "device": args.device,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "counts": {
                    "obstacles": args.obstacles,
                    "spawns": args.spawns,
                    "cores": args.cores,
                },
                "candidate_budgets": budgets,
                "ordinary_uninstrumented": timing,
                "untraced_outcome": {
                    "error_type": untraced["error_type"],
                    "error_message": untraced["error_message"],
                },
                "instrumented_outcome": {
                    "error_type": traced["error_type"],
                    "error_message": traced["error_message"],
                    "attempt_counts": traced["attempt_counts"],
                },
                "instrumentation_equivalence": _same_outcome(
                    untraced, traced
                ),
            }

            print(json.dumps(record, sort_keys=True), flush=True)

            if not record["instrumentation_equivalence"]:
                return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
