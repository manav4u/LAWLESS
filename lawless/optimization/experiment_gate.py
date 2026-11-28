from __future__ import annotations


import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()




def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)

def _utc(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Invalid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return parsed.astimezone(timezone.utc)




def _commit(root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root,
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()

    except (OSError, subprocess.CalledProcessError):
        return None




def _deny(code: str, detail: str) -> dict[str, Any]:
    return {
        "decision": "denied",
        "authorized": False,
        "failures": [{"code": code, "detail": detail}],
    }

def _report_ok(path: str | Path, label: str) -> tuple[bool, str]:
    try:
        report = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"{label} report unreadable: {exc}"
    if not isinstance(report, dict) or report.get("valid") is not True:
        return False, f"{label} report does not declare valid=true"
    return True, ""

def _verify_c3(
    provenance_path: str | Path,
    repo_root: str | Path,
    evidence_dir: str | Path,
    source_files: tuple[str, ...] | None,
) -> tuple[bool, str]:
    from lawless.optimization.provenance import verify_provenance


    report = verify_provenance(
        provenance_path, repo_root, evidence_dir, source_files
    )

    if report.get("valid") is not True:
        return False, "C3 provenance verification failed: " + json.dumps(
            report.get("failures", []), sort_keys=True
        )
    return True, ""




def _manifest_identity(evidence_dir: str | Path) -> tuple[dict, str]:
    manifest = _read_json(Path(evidence_dir) / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("C1 manifest must be a JSON object")
    return manifest, _canonical_hash(manifest)

def authorize_experiment(
    *,
    repo_root: str | Path,
    evidence_dir: str | Path,
    provenance_path: str | Path,
    c2_report_path: str | Path,
    authorization_path: str | Path,
    requested_budget: int,
    now: datetime | None = None,
    max_age_seconds: int = 86400,
    source_files: tuple[str, ...] | None = None,
) -> dict[str, Any]:

    supplied_now = now or datetime.now(timezone.utc)
    if supplied_now.tzinfo is None or supplied_now.utcoffset() is None:
        return _deny("invalid_clock", "Evaluation time must be timezone-aware")
    checked_at = supplied_now.astimezone(timezone.utc)
    if requested_budget <= 0 or max_age_seconds <= 0:
        return _deny("invalid_budget_or_age",
                     "Budget and authorization age must be positive")

    try:
        manifest, identity = _manifest_identity(evidence_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _deny("manifest_invalid", str(exc))


    c2_ok, c2_error = _report_ok(c2_report_path, "C2")
    if not c2_ok:
        return _deny("c2_validation_failed", c2_error)

    c3_ok, c3_error = _verify_c3(
        provenance_path, repo_root, evidence_dir, source_files
    )
    if not c3_ok:
        return _deny("c3_integrity_failed", c3_error)


    checkout_commit = _commit(repo_root)
    if not checkout_commit:
        return _deny("checkout_commit_unavailable",
                     "Cannot identify current Git commit")

    try:
        approval = _read_json(authorization_path)
    except (OSError, json.JSONDecodeError) as exc:
        return _deny("authorization_unreadable", str(exc))


    if not isinstance(approval, dict):
        return _deny("authorization_malformed",
                     "Authorization must be a JSON object")

    required = {
        "schema_version", "authorized", "issued_utc", "expires_utc",
        "checkout_commit", "experiment_manifest_sha256",
        "max_candidate_evaluations",
    }
    missing = sorted(required - approval.keys())
    if missing:
        return _deny("authorization_fields_missing", ", ".join(missing))

    if approval["schema_version"] != SCHEMA_VERSION:
        return _deny("authorization_schema", "Unsupported schema version")
    if approval["authorized"] is not True:
        return _deny("not_authorized", "Explicit authorization is required")
    if approval["checkout_commit"] != checkout_commit:
        return _deny("checkout_commit_mismatch",
                     "Approval is for a different checkout commit")

    if approval["experiment_manifest_sha256"] != identity:
        return _deny("experiment_identity_mismatch",
                     "Approval is for a different C1 manifest")
    limit = approval["max_candidate_evaluations"]
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        return _deny("authorization_budget_invalid",
                     "Authorized budget must be a positive integer")
    if requested_budget > limit:
        return _deny("budget_exceeded",
                     f"Requested {requested_budget}; authorized {limit}")

    try:
        issued = _utc(approval["issued_utc"])
        expires = _utc(approval["expires_utc"])
    except (TypeError, ValueError) as exc:
        return _deny("authorization_timestamp_invalid", str(exc))


    if expires <= issued:
        return _deny("authorization_interval_invalid",
                     "Expiry must follow issue time")

    if checked_at < issued or checked_at >= expires:
        return _deny("authorization_expired",
                     "Authorization is not currently within its validity window")
    if (checked_at - issued).total_seconds() > max_age_seconds:
        return _deny("authorization_stale",
                     "Authorization exceeds the permitted maximum age")

    return {
        "decision": "authorized",
        "authorized": True,
        "checked_utc": checked_at.isoformat(),
        "checkout_commit": checkout_commit,
        "experiment_manifest_sha256": identity,
        "requested_candidate_evaluations": requested_budget,
        "authorized_candidate_evaluations":
            approval["max_candidate_evaluations"],
        "authorization_expires_utc": expires.isoformat(),
        "dry_run": True,
        "experiment_launched": False,
        "failures": [],
    }

def inspect_experiment(
    **kwargs: Any,
) -> dict[str, Any]:
    """Inspect authorization only; never starts an experiment."""
    result = authorize_experiment(**kwargs)
    result["dry_run"] = True
    result["experiment_launched"] = False
    return result

def write_gate_record(
    record: dict[str, Any], output_path: str | Path,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
