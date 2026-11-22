"""Supplemental regression tests for LAWLESS P1.8-A A1 benchmark.

These tests do not modify production code, P0, or the existing focused
benchmark test module.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmarks import environment_benchmark as bench


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Validation boundary
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        0,
        -1,
        "1.0",
        None,
    ],
)
def test_wall_budget_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        bench.validate_wall_budget(value)


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        True,
        False,
        1.0,
        2.5,
        "2",
        None,
    ],
)
def test_step_bound_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        bench.validate_step_bound(value)


@pytest.mark.parametrize("value", [1, 2, 10, 1000])
def test_step_bound_accepts_positive_integers(value):
    bench.validate_step_bound(value)


@pytest.mark.parametrize(
    "value",
    [
        1,
        0.1,
        0.000001,
        100.0,
    ],
)
def test_wall_budget_accepts_finite_positive_values(value):
    bench.validate_wall_budget(value)


@pytest.mark.parametrize(
    "seed,repetitions",
    [
        (-1, 1),
        (True, 1),
        (1.0, 1),
        (0, 0),
        (0, False),
        (bench.MAX_SUPPORTED_SEED, 2),
        (bench.MAX_SUPPORTED_SEED + 1, 1),
    ],
)
def test_seed_span_rejects_invalid_values(seed, repetitions):
    with pytest.raises(ValueError):
        bench.validate_seed_span(seed, repetitions)


@pytest.mark.parametrize(
    "seed,repetitions",
    [
        (0, 1),
        (7, 2),
        (100, 10),
        (bench.MAX_SUPPORTED_SEED, 1),
    ],
)
def test_seed_span_accepts_supported_ranges(seed, repetitions):
    bench.validate_seed_span(seed, repetitions)


def test_execute_run_rejects_invalid_inputs_before_receipt(tmp_path):
    output = tmp_path / "should_not_exist.json"

    with pytest.raises(ValueError):
        bench.execute_run(
            root=REPO_ROOT,
            mode="smoke",
            batch=1,
            candidates=1,
            seed=7,
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_steps=True,
            post_reset_wall_budget=1.0,
            output_path=output,
        )

    assert not output.exists()


def test_execute_run_rejects_nan_budget_before_receipt(tmp_path):
    output = tmp_path / "should_not_exist.json"

    with pytest.raises(ValueError):
        bench.execute_run(
            root=REPO_ROOT,
            mode="smoke",
            batch=1,
            candidates=1,
            seed=7,
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_steps=1,
            post_reset_wall_budget=float("nan"),
            output_path=output,
        )

    assert not output.exists()


@pytest.mark.parametrize("device_type", ["meta", "mps", "xpu"])
def test_unsupported_device_types_are_rejected(device_type):
    with pytest.raises(ValueError, match="unsupported benchmark device"):
        bench.validate_device(torch.device(device_type))


def test_cpu_device_validation():
    bench.validate_device(torch.device("cpu"))


def test_cpu_index_is_rejected():
    with pytest.raises(ValueError, match="CPU device"):
        bench.validate_device(torch.device("cpu:0"))


def test_cuda_unavailable_is_rejected_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(
        ValueError,
        match=r"CUDA requested but CUDA is unavailable",
    ):
        bench.validate_device(torch.device("cuda"))


# ---------------------------------------------------------------------------
# CLI validation and argument failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "arguments",
    [
        ["--post-reset-wall-budget", "nan"],
        ["--post-reset-wall-budget", "inf"],
        ["--post-reset-wall-budget", "-inf"],
        ["--post-reset-wall-budget", "0"],
        ["--max-steps", "0"],
        ["--max-steps", "-1"],
        ["--repetitions", "0"],
        ["--batch", "0"],
        ["--candidates", "0"],
        ["--seed", "-1"],
    ],
)
def test_cli_invalid_values_exit_normally(arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["environment_benchmark", *arguments])

    with pytest.raises(SystemExit) as exc:
        bench.parse_args()

    assert exc.value.code == 2


def test_cli_seed_span_rejected(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "environment_benchmark",
            "--seed",
            str(bench.MAX_SUPPORTED_SEED),
            "--repetitions",
            "2",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        bench.parse_args()

    assert exc.value.code == 2


def test_cli_unsupported_device_rejected_before_output(monkeypatch):
    output = "regression_cli_output"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "environment_benchmark",
            "--device",
            "meta",
            "--output-dir",
            output,
        ],
    )

    with pytest.raises(SystemExit) as exc:
        bench.parse_args()

    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Real CLI integration
# ---------------------------------------------------------------------------

def run_cli(cwd: Path, *arguments: str):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.environment_benchmark",
            *arguments,
        ],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def test_real_cli_seed_repetitions_and_output_preservation(tmp_path):
    output_dir = tmp_path / "results"

    # Existing invocation data must survive. CLI creates distinct invocation
    # directories rather than replacing earlier receipts.
    output_dir.mkdir()
    marker = output_dir / "existing_marker.txt"
    marker.write_text("preserve\n", encoding="utf-8")

    result = run_cli(
        REPO_ROOT,
        "--mode",
        "smoke",
        "--device",
        "cpu",
        "--batch",
        "1",
        "--candidates",
        "4",
        "--repetitions",
        "2",
        "--max-steps",
        "1",
        "--post-reset-wall-budget",
        "60",
        "--seed",
        "7",
        "--output-dir",
        str(output_dir),
    )

    assert result.returncode == 2, result.stderr
    assert marker.read_text(encoding="utf-8") == "preserve\n"

    invocation_dirs = [
        item
        for item in output_dir.iterdir()
        if item.is_dir()
    ]

    assert len(invocation_dirs) == 1

    invocation_dir = invocation_dirs[0]
    receipt_path = invocation_dir / "invocation.json"

    assert receipt_path.is_file()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert receipt["status"] == "bounded_incomplete"
    assert receipt["overall_p1_8_a_status"] == "INCOMPLETE"
    assert receipt["workload"]["seed"] == 7
    assert receipt["workload"]["repetitions"] == 2

    runs = receipt["runs"]

    assert len(runs) == 2
    assert [entry["seed"] for entry in runs] == [7, 8]
    assert [entry["run_file"] for entry in runs] == [
        "run_000.json",
        "run_001.json",
    ]

    for index, expected_seed in enumerate((7, 8)):
        run_path = invocation_dir / f"run_{index:03d}.json"
        assert run_path.is_file()

        run = json.loads(run_path.read_text(encoding="utf-8"))

        assert run["status"] == "bounded_incomplete"
        assert run["root_seed_applied"] == expected_seed
        assert run["seed_requested"] == expected_seed
        assert len(run["actual_world_seeds"]) == 1
        assert len(run["geometry_inventory"]) == 1
        assert run["geometry_inventory"][0]["obstacle_count"] > 0
        assert run["geometry_inventory"][0]["core_count"] > 0
        assert run["accounting"]["world_step_transitions"] == 1
        assert run["accounting"]["candidate_slot_transitions"] == 4
        assert run["accounting"]["alive_agent_transitions"] == 4

    # Separate repetition seeds must be reflected in actual environment
    # world-seed receipts, not only the requested seed field.
    run0 = json.loads(
        (invocation_dir / "run_000.json").read_text(encoding="utf-8")
    )
    run1 = json.loads(
        (invocation_dir / "run_001.json").read_text(encoding="utf-8")
    )

    assert run0["actual_world_seeds"] != run1["actual_world_seeds"]


@pytest.mark.parametrize(
    "arguments",
    [
        ("--post-reset-wall-budget", "nan"),
        ("--post-reset-wall-budget", "inf"),
        ("--max-steps", "0"),
        ("--repetitions", "0"),
        ("--seed", "-1"),
        ("--device", "meta"),
    ],
)
def test_invalid_cli_does_not_create_output_directory(
    tmp_path,
    arguments,
):
    output = tmp_path / "invalid_output"

    result = run_cli(
        REPO_ROOT,
        "--output-dir",
        str(output),
        *arguments,
    )

    assert result.returncode == 2
    assert not output.exists()


# ---------------------------------------------------------------------------
# Global RNG isolation
# ---------------------------------------------------------------------------

def test_bounded_real_run_does_not_change_global_torch_rng(tmp_path):
    incoming_rng_state = torch.random.get_rng_state().clone()

    try:
        torch.manual_seed(20261122)
        state_before = torch.random.get_rng_state().clone()

        output = tmp_path / "rng_run.json"

        run = bench.execute_run(
            root=REPO_ROOT,
            mode="smoke",
            batch=1,
            candidates=4,
            seed=7,
            device=torch.device("cpu"),
            dtype=torch.float32,
            max_steps=1,
            post_reset_wall_budget=60.0,
            output_path=output,
        )

        state_after = torch.random.get_rng_state()

        assert torch.equal(state_before, state_after)
        assert run["status"] == "bounded_incomplete"
        assert run["accounting"]["world_step_transitions"] == 1
        assert output.is_file()

    finally:
        torch.random.set_rng_state(incoming_rng_state)


# ---------------------------------------------------------------------------
# Raw Git blob hashing
# ---------------------------------------------------------------------------

def git_repo(tmp_path: Path) -> Path:
    root = tmp_path / "gitrepo"
    root.mkdir()

    subprocess.run(
        ["git", "init"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ["git", "config", "user.email", "regression@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Regression Test"],
        cwd=root,
        check=True,
    )

    return root


def git_commit(root: Path, message: str):
    subprocess.run(
        ["git", "add", "--all"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_frozen_tag_hash_uses_raw_crlf_and_non_ascii_blob(tmp_path):
    root = git_repo(tmp_path)

    source = root / "lawless" / "raw_fixture.py"
    source.parent.mkdir(parents=True)

    raw = (
        "# CRLF and UTF-8 fixture\r\n"
        "message = 'café - 測試'\r\n"
    ).encode("utf-8")

    source.write_bytes(raw)
    git_commit(root, "fixture")

    subprocess.run(
        ["git", "tag", bench.FROZEN_P0_TAG],
        cwd=root,
        check=True,
    )

    result = bench.frozen_tag_python_inventory(root)

    assert result["status"] == "verified"
    assert result["tag"] == bench.FROZEN_P0_TAG
    assert result["tag_commit"]

    entry = result["entries"]["lawless/raw_fixture.py"]

    assert entry["status"] == "verified"
    assert entry["sha256"] == hashlib.sha256(raw).hexdigest()
    assert entry["size_bytes"] == len(raw)


def test_frozen_tag_commit_id_matches_actual_tag_commit(tmp_path):
    root = git_repo(tmp_path)

    source = root / "fixture.py"
    source.write_text("value = 1\n", encoding="utf-8")

    git_commit(root, "fixture")

    subprocess.run(
        ["git", "tag", bench.FROZEN_P0_TAG],
        cwd=root,
        check=True,
    )

    expected = subprocess.run(
        ["git", "rev-parse", "--verify", f"{bench.FROZEN_P0_TAG}^{{commit}}"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()

    result = bench.frozen_tag_python_inventory(root)

    assert result["tag_commit"] == expected


# ---------------------------------------------------------------------------
# Source identity failure and status
# ---------------------------------------------------------------------------

def test_frozen_identity_is_unknown_without_git_tag(tmp_path):
    root = git_repo(tmp_path)

    result = bench.frozen_tag_python_inventory(root)

    assert result["status"] == "unknown"
    assert result["tag_commit"] is None


def test_nonrepository_dirty_status_is_not_falsely_dirty(tmp_path):
    root = tmp_path / "not_a_repository"
    root.mkdir()

    result = bench.git_dirty_status(root)

    assert result["status"] in ("unknown", "error")
    assert result["dirty"] is None


def test_source_identity_propagates_incomplete_frozen_inventory(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "fake_repo"
    root.mkdir()

    monkeypatch.setattr(
        bench,
        "tracked_python_inventory",
        lambda _root: ["fixture.py"],
    )
    monkeypatch.setattr(
        bench,
        "untracked_python_inventory",
        lambda _root: [],
    )
    monkeypatch.setattr(
        bench,
        "git_dirty_status",
        lambda _root: {
            "status": "verified",
            "dirty": False,
            "error": None,
        },
    )
    monkeypatch.setattr(
        bench,
        "git_success",
        lambda _root, *_args: "abc123",
    )
    monkeypatch.setattr(
        bench,
        "file_hash_inventory",
        lambda _root, _paths: {
            "status": "verified",
            "entries": {
                "fixture.py": {
                    "status": "verified",
                    "sha256": "abc",
                },
                "benchmarks/environment_benchmark.py": {
                    "status": "verified",
                    "sha256": "def",
                },
            },
            "missing_or_error": [],
        },
    )
    monkeypatch.setattr(
        bench,
        "frozen_tag_python_inventory",
        lambda _root: {
            "status": "unknown",
            "tag": bench.FROZEN_P0_TAG,
            "tag_commit": None,
            "error": "tag missing",
            "entries": {},
        },
    )

    result = bench.source_identity(root)

    assert result["status"] == "incomplete"
    assert result["identity_complete"] is False
    assert result["frozen_p0_python_hashes"]["status"] == "unknown"


def test_untracked_source_identity_inventory_is_explicit():
    inventory = bench.untracked_python_inventory(REPO_ROOT)

    assert inventory is not None
    assert isinstance(inventory, list)


# ---------------------------------------------------------------------------
# Runtime metadata path exposure
# ---------------------------------------------------------------------------

def test_runtime_metadata_omits_executable_and_workspace_paths():
    metadata = bench.runtime_metadata(torch.device("cpu"))

    assert "python_executable" not in metadata
    assert "processor" not in metadata
    assert "platform" not in metadata

    assert "python" in metadata
    assert "python_version" in metadata
    assert "torch_version" in metadata
    assert "torch_cuda_version" in metadata
    assert "device" in metadata
    assert "cuda_available" in metadata


def test_runtime_metadata_reports_supported_versions():
    metadata = bench.runtime_metadata(torch.device("cpu"))

    assert metadata["python_version"] == sys.version.split()[0]
    assert metadata["torch_version"] == str(torch.__version__)
    assert metadata["device"] == "cpu"


# ---------------------------------------------------------------------------
# Accounting scope and receipt behavior
# ---------------------------------------------------------------------------

def test_accounted_rate_scope_is_not_uninstrumented_throughput():
    run = {
        "accounting": {
            "alive_agent_transitions": 8,
            "candidate_slot_transitions": 8,
            "world_step_transitions": 2,
        },
        "elapsed_post_reset_wall_seconds": 1.0,
        "step_call_samples_seconds": [0.1, 0.2],
    }

    summary = bench.run_summary(run)

    assert summary["accounted_rates"]["rate_scope"] == (
        "accounted_bounded_A1"
    )
    assert (
        summary["accounted_rates"]
        ["alive_agent_transitions_per_post_reset_wall_second"]
        == 8
    )


def test_atomic_receipt_write_preserves_valid_json(tmp_path):
    receipt_path = tmp_path / "receipt.json"

    bench.atomic_write_json(
        receipt_path,
        {"status": "partial", "step": 1},
    )

    first = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert first == {"status": "partial", "step": 1}

    bench.atomic_write_json(
        receipt_path,
        {"status": "partial", "step": 2},
    )

    second = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert second == {"status": "partial", "step": 2}
    assert not receipt_path.with_suffix(".json.tmp").exists()

def test_cli_malformed_device_string_uses_argparse_error(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "environment_benchmark",
            "--device",
            "not-a-device",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        bench.parse_args()

    assert exc.value.code == 2


def test_cli_invalid_cuda_index_uses_argparse_error(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: 1,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "environment_benchmark",
            "--device",
            "cuda:1",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        bench.parse_args()

    assert exc.value.code == 2


def test_cli_malformed_device_does_not_create_output_directory(tmp_path):
    output = tmp_path / "malformed_device_output"

    result = run_cli(
        REPO_ROOT,
        "--device",
        "not-a-device",
        "--output-dir",
        str(output),
    )

    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()
