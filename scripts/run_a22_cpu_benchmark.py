#!/usr/bin/env python3
"""Bounded supervisor for the byte-preserved LAWLESS A2.2 CPU cell.

The original cell remains unchanged on disk. This supervisor:
- verifies its exact SHA-256;
- makes a temporary execution copy with narrowly scoped stage logging
  and timeouts for the cell's own git-command helper;
- runs that copy as one Python process;
- enforces a wall-clock timeout over the entire process group;
- collects only the cell's evidence directory and evidence archive;
- never dumps the environment or captures credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


EXPECTED_CELL_SHA256 = (
    "5c34a9f3fc7d4d0f8463a3dbc70e375"
    "44776fd4d44aecfe18f542b53231bee51"
)

CELL_RELATIVE_PATH = Path(
    "scripts/lawless_p18_a22_colab_cell_candidate.py"
)

CONTENT_ROOT = Path("/content")
WORKSPACE_GLOB = "lawless_a22_colab*"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    path.write_text(payload, encoding="utf-8")


class SupervisorLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.stream = path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        line = f"[{utc_now()}] {message}"
        self.stream.write(line + "\n")
        self.stream.flush()
        print(line, flush=True)

    def close(self) -> None:
        self.stream.close()


def ensure_content_directory(log: SupervisorLog) -> None:
    """Provide the /content path required by the unmodified cell."""
    if not CONTENT_ROOT.exists():
        log.write("Creating /content for the original cell.")
        subprocess.run(
            ["sudo", "mkdir", "-p", str(CONTENT_ROOT)],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    if not os.access(CONTENT_ROOT, os.W_OK | os.X_OK):
        log.write("Adjusting /content ownership for the runner user.")
        subprocess.run(
            [
                "sudo",
                "chown",
                f"{os.getuid()}:{os.getgid()}",
                str(CONTENT_ROOT),
            ],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

    if not os.access(CONTENT_ROOT, os.W_OK | os.X_OK):
        raise RuntimeError("/content is not writable by the runner user.")


def replace_once(
    source: str,
    old: str,
    new: str,
    description: str,
) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Observability patch anchor {description!r} occurred "
            f"{count} times; expected exactly once. Refusing to run."
        )
    return source.replace(old, new, 1)


def make_observable_execution_copy(
    original_path: Path,
    temporary_directory: Path,
) -> tuple[Path, str, list[str]]:
    """Patch only a temporary copy; never rewrite the tracked source."""
    original_bytes = original_path.read_bytes()
    actual_hash = hashlib.sha256(original_bytes).hexdigest()

    if actual_hash != EXPECTED_CELL_SHA256:
        raise RuntimeError(
            "Operational cell SHA-256 mismatch: "
            f"actual={actual_hash}, expected={EXPECTED_CELL_SHA256}"
        )

    source = original_bytes.decode("utf-8")
    patches: list[str] = []

    helper_anchor = (
        "# The one exact source-checkpoint SHA for this run.\n"
    )
    helper_replacement = (
        "def _a22_supervisor_stage(label):\n"
        "    print(f\"[A2.2 stage] {label}\", flush=True)\n"
        "\n"
        "\n"
        + helper_anchor
    )
    source = replace_once(
        source,
        helper_anchor,
        helper_replacement,
        "stage helper insertion",
    )
    patches.append("insert temporary stage logger")

    old_run_command = (
        "        stderr=subprocess.PIPE,\n"
        "        check=False,\n"
        "    )\n"
    )
    new_run_command = (
        "        stderr=subprocess.PIPE,\n"
        "        check=False,\n"
        "        timeout=300,\n"
        "    )\n"
    )
    source = replace_once(
        source,
        old_run_command,
        new_run_command,
        "cell git-command timeout",
    )
    patches.append("add 300-second timeout to cell run_command")

    source = replace_once(
        source,
        "try:\n"
        "    # Clone only the public validation branch, then detach at the exact commit.\n",
        "try:\n"
        "    _a22_supervisor_stage('clone_and_pin_start')\n"
        "    # Clone only the public validation branch, then detach at the exact commit.\n",
        "clone start",
    )
    patches.append("log clone and pin start")

    source = replace_once(
        source,
        "    initial_tree_status = git_clean_status(repo)\n",
        "    initial_tree_status = git_clean_status(repo)\n"
        "    _a22_supervisor_stage('clone_and_pin_complete')\n",
        "clone completion",
    )
    patches.append("log clone and pin completion")

    source = replace_once(
        source,
        "    # Capture actual runtime/thread configuration; do not change it.\n",
        "    _a22_supervisor_stage('runtime_capture_start')\n"
        "    # Capture actual runtime/thread configuration; do not change it.\n",
        "runtime capture start",
    )
    patches.append("log runtime capture start")

    source = replace_once(
        source,
        "    # Validate source identity/frozen P0 before starting either workload.\n",
        "    _a22_supervisor_stage('runtime_capture_complete')\n"
        "    _a22_supervisor_stage('frozen_source_preflight_start')\n"
        "    # Validate source identity/frozen P0 before starting either workload.\n",
        "runtime capture completion and frozen preflight start",
    )
    patches.append("log runtime capture completion and frozen preflight start")

    source = replace_once(
        source,
        '    verify_frozen_identity(initial_identity, SOURCE_CHECKPOINT, "Initial checkout")\n',
        '    verify_frozen_identity(initial_identity, SOURCE_CHECKPOINT, "Initial checkout")\n'
        "    _a22_supervisor_stage('frozen_source_preflight_complete')\n",
        "frozen preflight completion",
    )
    patches.append("log frozen source preflight completion")

    source = replace_once(
        source,
        "    profile_receipt = profiler_module.profile_episode(\n",
        "    _a22_supervisor_stage('component_profile_start')\n"
        "    profile_receipt = profiler_module.profile_episode(\n",
        "profile start",
    )
    patches.append("log component profile start")

    source = replace_once(
        source,
        "        output_path=profile_path,\n"
        "    )\n"
        '    require(profile_path.is_file(), "Profiler did not persist its raw receipt")\n',
        "        output_path=profile_path,\n"
        "    )\n"
        "    _a22_supervisor_stage('component_profile_returned')\n"
        '    require(profile_path.is_file(), "Profiler did not persist its raw receipt")\n',
        "profile return",
    )
    patches.append("log component profile return")

    source = replace_once(
        source,
        "    timing_receipt = timing_module.run_benchmark(\n",
        "    _a22_supervisor_stage('timing_run_start')\n"
        "    timing_receipt = timing_module.run_benchmark(\n",
        "timing run start",
    )
    patches.append("log uninstrumented timing start")

    source = replace_once(
        source,
        "        repetitions=TIMING_REPETITIONS,\n"
        "    )\n"
        "    timing_module.write_receipt(timing_receipt, timing_path)\n",
        "        repetitions=TIMING_REPETITIONS,\n"
        "    )\n"
        "    _a22_supervisor_stage('timing_run_returned')\n"
        "    timing_module.write_receipt(timing_receipt, timing_path)\n",
        "timing run return",
    )
    patches.append("log uninstrumented timing return")

    source = replace_once(
        source,
        "    # Timing must not have left wrappers installed or contaminated the process.\n",
        "    _a22_supervisor_stage('timing_receipt_validated')\n"
        "    # Timing must not have left wrappers installed or contaminated the process.\n",
        "timing receipt validation completion",
    )
    patches.append("log timing receipt validation completion")

    source = replace_once(
        source,
        "except BaseException as exc:\n"
        "    failure = {\n",
        "except BaseException as exc:\n"
        "    _a22_supervisor_stage('cell_exception_captured')\n"
        "    failure = {\n",
        "cell exception capture",
    )
    patches.append("log cell exception capture")

    source = replace_once(
        source,
        "finally:\n"
        "    # Preserve any returned raw receipts, including invalid receipts.\n",
        "finally:\n"
        "    _a22_supervisor_stage('evidence_finalization_start')\n"
        "    # Preserve any returned raw receipts, including invalid receipts.\n",
        "evidence finalization start",
    )
    patches.append("log evidence finalization start")

    source = replace_once(
        source,
        'with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:\n',
        "_a22_supervisor_stage('evidence_packaging_start')\n"
        'with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:\n',
        "evidence packaging start",
    )
    patches.append("log evidence packaging start")

    runtime_path = temporary_directory / "a22_cell_observable_runtime_copy.py"
    runtime_bytes = source.encode("utf-8")
    runtime_path.write_bytes(runtime_bytes)

    runtime_hash = hashlib.sha256(runtime_bytes).hexdigest()
    return runtime_path, runtime_hash, patches


def terminate_process_group(process: subprocess.Popen[str]) -> None:
    """Terminate the cell and any git subprocesses it started."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass

    # Also clear any child process that outlived the Python parent.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    process.wait()


def run_cell(
    runtime_cell: Path,
    repository_root: Path,
    cell_log_path: Path,
    timeout_seconds: int,
    log: SupervisorLog,
) -> tuple[int, bool]:
    cell_log_path.parent.mkdir(parents=True, exist_ok=True)
    child_environment = os.environ.copy()
    child_environment["PYTHONUNBUFFERED"] = "1"

    log.write(
        f"Starting one cell process; wall-clock limit={timeout_seconds}s."
    )

    process = subprocess.Popen(
        [sys.executable, "-u", str(runtime_cell)],
        cwd=str(repository_root),
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )

    def forward_output() -> None:
        assert process.stdout is not None
        with cell_log_path.open("w", encoding="utf-8") as stream:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)

    output_thread = threading.Thread(
        target=forward_output,
        name="a22-cell-output",
        daemon=True,
    )
    output_thread.start()

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        log.write(
            "Cell wall-clock timeout reached; terminating its process group."
        )
        terminate_process_group(process)

    output_thread.join(timeout=15)
    if output_thread.is_alive():
        log.write("Warning: cell output reader did not finish within 15 seconds.")

    exit_code = 124 if timed_out else int(process.returncode or 0)
    log.write(
        f"Cell process finished: exit_code={exit_code}, timed_out={timed_out}."
    )
    return exit_code, timed_out


def copy_evidence(
    content_root: Path,
    previous_workspaces: set[Path],
    artifact_root: Path,
) -> list[dict[str, object]]:
    """Copy only evidence files, never cloned checkouts or whole workspaces."""
    copied: list[dict[str, object]] = []
    raw_root = artifact_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    if not content_root.is_dir():
        return copied

    for workspace in sorted(content_root.glob(WORKSPACE_GLOB)):
        try:
            resolved_workspace = workspace.resolve()
        except OSError:
            continue

        if resolved_workspace in previous_workspaces or not workspace.is_dir():
            continue

        destination_root = raw_root / workspace.name
        evidence_dir = workspace / "evidence"

        if evidence_dir.is_dir():
            for source_file in sorted(evidence_dir.rglob("*")):
                if not source_file.is_file() or source_file.is_symlink():
                    continue

                relative = source_file.relative_to(evidence_dir)
                destination = destination_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination)

                copied.append(
                    {
                        "path": destination.relative_to(artifact_root).as_posix(),
                        "size_bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination),
                    }
                )

        cell_archive = workspace / "LAWLESS_A22_CPU_evidence.zip"
        if cell_archive.is_file() and not cell_archive.is_symlink():
            destination = destination_root / cell_archive.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cell_archive, destination)
            copied.append(
                {
                    "path": destination.relative_to(artifact_root).as_posix(),
                    "size_bytes": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                }
            )

    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("artifacts/a22-cpu"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be positive")

    repository_root = Path(__file__).resolve().parents[1]
    artifact_root = args.artifact_dir
    if not artifact_root.is_absolute():
        artifact_root = repository_root / artifact_root
    artifact_root.mkdir(parents=True, exist_ok=True)

    log = SupervisorLog(artifact_root / "supervisor.log")
    status: dict[str, object] = {
        "schema": "lawless.a22.cpu_supervisor_status.v1",
        "started_utc": utc_now(),
        "source_checkpoint": "9b9c3b28a41037671b0806d234e1d728082a9b28",
        "expected_cell_sha256": EXPECTED_CELL_SHA256,
        "timeout_seconds": args.timeout_seconds,
        "timed_out": False,
        "cell_exit_code": None,
        "runtime_cell_sha256": None,
        "observability_patches": [],
        "captured_evidence_files": [],
        "status": "failed",
        "error": None,
    }

    previous_workspaces: set[Path] = set()
    exit_code = 1

    try:
        cell_path = repository_root / CELL_RELATIVE_PATH
        if not cell_path.is_file():
            raise RuntimeError(f"Operational cell is missing: {cell_path}")

        actual_cell_hash = sha256_file(cell_path)
        status["actual_cell_sha256"] = actual_cell_hash
        if actual_cell_hash != EXPECTED_CELL_SHA256:
            raise RuntimeError(
                "Operational cell SHA-256 mismatch: "
                f"actual={actual_cell_hash}, expected={EXPECTED_CELL_SHA256}"
            )

        log.write("Verified exact original operational-cell SHA-256.")
        ensure_content_directory(log)

        previous_workspaces = {
            path.resolve()
            for path in CONTENT_ROOT.glob(WORKSPACE_GLOB)
            if path.is_dir()
        }

        with tempfile.TemporaryDirectory(
            prefix="lawless-a22-supervisor-"
        ) as temporary_name:
            temporary_directory = Path(temporary_name)
            runtime_cell, runtime_hash, patches = (
                make_observable_execution_copy(
                    cell_path,
                    temporary_directory,
                )
            )
            status["runtime_cell_sha256"] = runtime_hash
            status["observability_patches"] = patches

            log.write(
                "Prepared temporary observability copy; tracked source remains unchanged."
            )
            log.write(
                "The timing call itself has no injected progress output."
            )

            exit_code, timed_out = run_cell(
                runtime_cell=runtime_cell,
                repository_root=repository_root,
                cell_log_path=artifact_root / "cell.stdout-stderr.log",
                timeout_seconds=args.timeout_seconds,
                log=log,
            )
            status["timed_out"] = timed_out
            status["cell_exit_code"] = exit_code

        copied = copy_evidence(
            CONTENT_ROOT,
            previous_workspaces,
            artifact_root,
        )
        status["captured_evidence_files"] = copied

        if exit_code == 0:
            status["status"] = "cell_completed"
            log.write(
                "Cell exited successfully; its own receipt validators accepted the run."
            )
        elif status["timed_out"]:
            status["status"] = "timed_out"
            log.write(
                "Timeout: retained any evidence already persisted; no receipt was fabricated."
            )
        else:
            status["status"] = "cell_failed"
            log.write(
                "Cell failed; retained any raw evidence already persisted."
            )

    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["status"] = "supervisor_failed"
        log.write(f"Supervisor failure: {status['error']}")
        exit_code = 1

        # Preserve any evidence directory created before the supervisor error.
        try:
            copied = copy_evidence(
                CONTENT_ROOT,
                previous_workspaces,
                artifact_root,
            )
            status["captured_evidence_files"] = copied
        except BaseException as collection_error:
            log.write(
                "Evidence collection also failed: "
                f"{type(collection_error).__name__}: {collection_error}"
            )

    finally:
        status["finished_utc"] = utc_now()
        write_json(artifact_root / "supervisor_status.json", status)
        log.write(
            f"Supervisor result: {status['status']}; "
            f"captured_files={len(status['captured_evidence_files'])}."
        )
        log.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
