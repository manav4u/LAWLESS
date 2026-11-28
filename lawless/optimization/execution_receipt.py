from __future__ import annotations


import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"completed", "failed", "interrupted", "timed_out"}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def _canonical_hash(value: Any) -> str:
    data = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def _safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes evidence root: {relative}")
    return resolved




def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)

def _utc(value: str | datetime) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise ValueError("Timestamp must be a datetime or ISO-8601 string")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Timestamp must include timezone")
    return value.astimezone(timezone.utc)




def _git(root: Path, *args: str) -> str | None:

    try:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None




def _git_identity(root: Path) -> dict[str, str | None]:
    return {
        "commit": _git(root, "rev-parse", "HEAD"),
        "branch": _git(root, "branch", "--show-current"),
    }

def _runtime() -> dict[str, Any]:
    try:
        import torch
        torch_version = str(torch.__version__)
    except ImportError:
        torch_version = None
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "pytorch_version": torch_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "execution_device": "cpu",
    }

def _artifact_hashes(
    evidence: Path, artifact_paths: Iterable[str],
) -> list[dict[str, str]]:
    paths = list(artifact_paths)
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate artifact path")
    records = []
    for relative in paths:
        path = _safe_path(evidence, relative)
        if not path.is_file():
            raise FileNotFoundError(relative)

        records.append({
            "path": relative,
            "sha256": _sha256(path),
        })
    return records




def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value

def create_execution_receipt(
    *,
    repo_root: str | Path,
    evidence_dir: str | Path,
    authorization_path: str | Path,
    artifact_paths: Iterable[str],
    consumed_budget: int,
    started_utc: str | datetime,
    ended_utc: str | datetime,
    terminal_status: str,
    receipt_path: str | Path,
) -> dict[str, Any]:

    root = Path(repo_root).resolve()
    evidence = Path(evidence_dir).resolve()
    authorization_file = Path(authorization_path).resolve()
    manifest = _read_json(_safe_path(evidence, "manifest.json"))
    authorization = _read_json(authorization_file)
    if not isinstance(manifest, dict) or not isinstance(authorization, dict):
        raise ValueError("Manifest and authorization must be JSON objects")
    if authorization.get("authorized") is not True:
        raise ValueError("C4 authorization is not approved")

    authorized_budget = _positive_int(
        authorization.get("max_candidate_evaluations"),
        "Authorized budget",
    )
    consumed = _positive_int(consumed_budget, "Consumed budget")
    if consumed > authorized_budget:
        raise ValueError("Consumed budget exceeds C4 authorization")
    start = _utc(started_utc)
    end = _utc(ended_utc)
    if end < start:
        raise ValueError("End timestamp precedes start timestamp")

    if terminal_status not in TERMINAL_STATUSES:
        raise ValueError("Unsupported terminal status")
    manifest_hash = _canonical_hash(manifest)
    if authorization.get("experiment_manifest_sha256") != manifest_hash:
        raise ValueError("C4 authorization manifest identity mismatch")
    commit = _git_identity(root)
    if not commit["commit"] or not commit["branch"]:
        raise ValueError("Cannot identify checkout commit and branch")
    if authorization.get("checkout_commit") != commit["commit"]:
        raise ValueError("C4 authorization checkout commit mismatch")

    artifacts = _artifact_hashes(evidence, artifact_paths)
    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment_manifest_sha256": manifest_hash,
        "authorization_sha256": _sha256(authorization_file),
        "authorized_candidate_evaluations": authorized_budget,
        "consumed_candidate_evaluations": consumed,
        "started_utc": start.isoformat(),
        "ended_utc": end.isoformat(),
        "elapsed_seconds": (end - start).total_seconds(),

        "checkout": commit,
        "terminal_status": terminal_status,
        "artifacts": artifacts,
        "runtime": _runtime(),
    }
    target = Path(receipt_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return record

def _failure(
    failures: list[dict[str, str]], code: str, detail: str,
) -> None:
    failures.append({"code": code, "detail": detail})




def _load_record(path: str | Path) -> dict[str, Any]:
    value = _read_json(Path(path))
    if not isinstance(value, dict):
        raise ValueError("Receipt root must be a JSON object")
    return value

def verify_execution_receipt(
    *,
    receipt_path: str | Path,
    repo_root: str | Path,
    evidence_dir: str | Path,
    authorization_path: str | Path,
    c2_report_path: str | Path,
    provenance_path: str | Path,
    source_files: tuple[str, ...] | None = None,
) -> dict[str, Any]:

    failures: list[dict[str, str]] = []
    root = Path(repo_root).resolve()
    evidence = Path(evidence_dir).resolve()
    try:
        record = _load_record(receipt_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"valid": False, "failures": [{
            "code": "receipt_unreadable", "detail": str(exc)
        }]}

    required = {
        "schema_version", "experiment_manifest_sha256",
        "authorization_sha256", "authorized_candidate_evaluations",
        "consumed_candidate_evaluations", "started_utc", "ended_utc",
        "elapsed_seconds", "checkout", "terminal_status",
        "artifacts", "runtime",
    }
    for key in sorted(required - record.keys()):
        _failure(failures, "missing_field", key)

    if record.get("schema_version") != SCHEMA_VERSION:
        _failure(failures, "schema_version", "Unsupported receipt schema")
    try:
        manifest = _read_json(_safe_path(evidence, "manifest.json"))
        if _canonical_hash(manifest) != record.get(
            "experiment_manifest_sha256"
        ):
            _failure(failures, "manifest_identity_mismatch",
                     "C1 manifest hash differs")

    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _failure(failures, "manifest_unreadable", str(exc))
    try:
        auth_hash = _sha256(Path(authorization_path).resolve())
        if auth_hash != record.get("authorization_sha256"):
            _failure(failures, "authorization_identity_mismatch",
                     "C4 authorization file hash differs")
        approval = _read_json(Path(authorization_path))

        if approval.get("experiment_manifest_sha256") != record.get(
            "experiment_manifest_sha256"
        ):
            _failure(failures, "authorization_manifest_mismatch",
                     "C4 authorization targets another experiment")
        if approval.get("max_candidate_evaluations") != record.get(
            "authorized_candidate_evaluations"
        ):
            _failure(failures, "authorization_budget_mismatch",
                     "C4 authorized budget differs from receipt")
        checkout = record.get("checkout", {})
        if approval.get("checkout_commit") != checkout.get("commit"):
            _failure(failures, "authorization_commit_mismatch",
                     "C4 authorization targets another checkout")

    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _failure(failures, "authorization_unreadable", str(exc))


    return _verify_remaining(
        record, failures, root, evidence, c2_report_path,
        provenance_path, source_files,
    )

def _verify_remaining(
    record: dict[str, Any],
    failures: list[dict[str, str]],
    root: Path,
    evidence: Path,
    c2_report_path: str | Path,
    provenance_path: str | Path,
    source_files: tuple[str, ...] | None,
) -> dict[str, Any]:
    authorized = record.get("authorized_candidate_evaluations")
    consumed = record.get("consumed_candidate_evaluations")

    try:
        authorized = _positive_int(authorized, "Authorized budget")
        consumed = _positive_int(consumed, "Consumed budget")
        if consumed > authorized:
            _failure(failures, "budget_exceeded",
                     "Consumed budget exceeds authorized budget")
    except ValueError as exc:
        _failure(failures, "budget_invalid", str(exc))


    status = record.get("terminal_status")
    if status not in TERMINAL_STATUSES:
        _failure(failures, "terminal_status_invalid", str(status))

    try:
        start = _utc(record.get("started_utc"))
        end = _utc(record.get("ended_utc"))
        elapsed = record.get("elapsed_seconds")
        expected = (end - start).total_seconds()
        if expected < 0 or isinstance(elapsed, bool):
            raise ValueError("Invalid elapsed interval")

        if not isinstance(elapsed, (int, float)) or elapsed != expected:
            _failure(failures, "duration_mismatch",
                     "Elapsed duration differs from timestamps")
    except (TypeError, ValueError) as exc:
        _failure(failures, "timestamp_invalid", str(exc))


    checkout = record.get("checkout")
    if not isinstance(checkout, dict):
        _failure(failures, "checkout_invalid", "Expected checkout object")

    else:
        current = _git_identity(root)
        if checkout.get("commit") != current["commit"]:
            _failure(failures, "checkout_commit_mismatch",
                     "Receipt checkout commit differs")
        if checkout.get("branch") != current["branch"]:
            _failure(failures, "checkout_branch_mismatch",
                     "Receipt checkout branch differs")

    runtime = record.get("runtime")
    if not isinstance(runtime, dict):
        _failure(failures, "runtime_invalid", "Expected runtime object")
    elif runtime.get("execution_device") != "cpu":
        _failure(failures, "execution_device_invalid",
                 "Receipt must declare CPU execution")

    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list):
        artifacts = []
        _failure(failures, "artifacts_invalid", "Expected artifact list")
    seen: set[str] = set()
    for item in artifacts:
        if not isinstance(item, dict):
            _failure(failures, "artifact_record_invalid", repr(item))
            continue

        relative, digest = item.get("path"), item.get("sha256")
        if not isinstance(relative, str) or relative in seen:
            _failure(failures, "artifact_duplicate_or_invalid", repr(relative))
            continue
        seen.add(relative)
        try:
            actual = _sha256(_safe_path(evidence, relative))
            if actual != digest:
                _failure(failures, "artifact_hash_mismatch", relative)
        except (OSError, ValueError) as exc:
            _failure(failures, "artifact_missing_or_unsafe",
                     f"{relative}: {exc}")

    if record.get("terminal_status") == "completed":
        try:
            report = _read_json(Path(c2_report_path))
            if not isinstance(report, dict) or report.get("valid") is not True:
                _failure(failures, "c2_validation_failed",
                         "C2 report does not declare valid=true")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            _failure(failures, "c2_report_unreadable", str(exc))

        from lawless.optimization.provenance import verify_provenance
        try:
            c3 = verify_provenance(
                provenance_path, root, evidence, source_files
            )
            if c3.get("valid") is not True:
                _failure(failures, "c3_integrity_failed",
                         json.dumps(c3.get("failures", []), sort_keys=True))
        except (OSError, ValueError, TypeError) as exc:
            _failure(failures, "c3_verification_error", str(exc))

    return {
        "valid": not failures,
        "failures": failures,
        "checked_artifacts": len(artifacts),
        "terminal_status": record.get("terminal_status"),
    }

def write_validation_report(
    report: dict[str, Any], report_path: str | Path,
) -> None:
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
