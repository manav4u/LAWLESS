"""Explicit experiment acceptance decisions; never authorizes execution."""
from __future__ import annotations


import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = 1
DECISIONS = {"accepted", "rejected", "inconclusive"}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value

def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )




def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)

def _validate_timestamp(value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("timestamp_utc must be a nonempty ISO timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp_utc must include a timezone")
    if parsed.utcoffset().total_seconds() != 0:
        raise ValueError("timestamp_utc must use UTC")

def _verify_package(root: Path, manifest: Path, archive: Path) -> dict:
    from .evidence_package import verify_evidence_package


    result = verify_evidence_package(root, manifest, archive)
    _require(isinstance(result, dict), "Invalid C6 verifier result")
    _require(result.get("valid") is True, "C6 package verification failed")
    identity = result.get("identity")
    _require(isinstance(identity, dict), "C6 identity missing")
    return identity

def _validate_identity(identity: dict) -> dict:
    required = {
        "manifest_sha256", "checkout_commit", "checkout_branch",
        "terminal_status", "authorized_candidate_evaluations",
        "consumed_candidate_evaluations",
    }
    _require(required.issubset(identity), "Incomplete C6 identity")
    _require(bool(HASH_RE.fullmatch(identity["manifest_sha256"])),
             "Invalid experiment manifest identity")
    _require(isinstance(identity["checkout_commit"], str) and
             bool(identity["checkout_commit"]), "Missing checkout commit")

    _require(isinstance(identity["checkout_branch"], str) and
             bool(identity["checkout_branch"]), "Missing checkout branch")
    _require(identity["terminal_status"] == "completed",
             "Only completed executions can receive a decision")
    cap = identity["authorized_candidate_evaluations"]
    used = identity["consumed_candidate_evaluations"]
    _require(isinstance(cap, int) and not isinstance(cap, bool) and cap > 0,
             "Invalid authorized evaluation budget")
    _require(isinstance(used, int) and not isinstance(used, bool) and
             0 <= used <= cap, "Invalid consumed evaluation budget")
    return identity

def _decision_identity(record: dict) -> tuple:
    return (
        record.get("experiment_id"),
        record.get("package_manifest_sha256"),
        record.get("zip_sha256"),
    )




def _check_existing_records(record: dict, paths) -> None:
    current = _decision_identity(record)
    for path in paths:
        other = _read_json(Path(path))
        if other.get("schema_version") != SCHEMA_VERSION:
            continue

        if _decision_identity(other) == current:
            if other.get("decision") == record["decision"]:
                raise ValueError("Duplicate acceptance decision")
            raise ValueError("Conflicting acceptance decisions")




def create_acceptance_decision(
    evidence_root,
    package_manifest_path,
    zip_path,
    decision_record_path,
    verification_report_path,
    *,
    decision,
    decision_maker,
    timestamp_utc,
    rationale,
    existing_decision_paths=(),
):

    root, manifest, archive = map(
        Path, (evidence_root, package_manifest_path, zip_path)
    )
    record_path, report_path = map(
        Path, (decision_record_path, verification_report_path)
    )
    _require(decision in DECISIONS, "Invalid explicit decision")
    _require(isinstance(decision_maker, str) and decision_maker.strip(),
             "decision_maker is required")
    _require(isinstance(rationale, str) and rationale.strip(),
             "Nonempty rationale is required")
    _validate_timestamp(timestamp_utc)

    _require(record_path.resolve() != report_path.resolve(),
             "Decision and report paths must differ")
    identity = _validate_identity(_verify_package(root, manifest, archive))
    record = {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "decision_maker": decision_maker.strip(),
        "timestamp_utc": timestamp_utc,
        "rationale": rationale.strip(),
        "package_manifest_sha256": _sha256(manifest),
        "zip_sha256": _sha256(archive),
        "checkout_commit": identity["checkout_commit"],
        "experiment_id": identity["manifest_sha256"],
    }
    _check_existing_records(record, existing_decision_paths)
    _write_json(record_path, record)
    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": False,
        "errors": [],
        "decision_record_sha256": _sha256(record_path),
    }
    _write_json(report_path, report)
    report = verify_acceptance_decision(
        root, manifest, archive, record_path,
        verification_report_path=report_path,
        existing_decision_paths=existing_decision_paths,
    )
    _write_json(report_path, report)
    if not report["valid"]:
        raise ValueError("Acceptance decision failed independent verification")
    return report

def verify_acceptance_decision(
    evidence_root,
    package_manifest_path,
    zip_path,
    decision_record_path,
    *,
    verification_report_path,
    existing_decision_paths=(),
) -> dict:
    errors = []
    try:
        root, manifest, archive = map(
            Path, (evidence_root, package_manifest_path, zip_path)
        )
        record_path = Path(decision_record_path)
        record = _read_json(record_path)
        anchor = _read_json(Path(verification_report_path))
        _require(anchor.get("schema_version") == SCHEMA_VERSION,
                 "Verification anchor schema mismatch")
        _require(anchor.get("decision_record_sha256") == _sha256(record_path),
                 "Decision record integrity mismatch")

        expected = {
            "schema_version", "decision", "decision_maker", "timestamp_utc",
            "rationale", "package_manifest_sha256", "zip_sha256",
            "checkout_commit", "experiment_id",
        }
        _require(set(record) == expected, "Decision fields mismatch")
        _require(record["schema_version"] == SCHEMA_VERSION, "Schema mismatch")
        _require(record["decision"] in DECISIONS, "Invalid decision")
        _require(isinstance(record["decision_maker"], str) and
                 bool(record["decision_maker"].strip()), "Missing decision-maker")
        _require(isinstance(record["rationale"], str) and
                 bool(record["rationale"].strip()), "Missing rationale")
        _validate_timestamp(record["timestamp_utc"])
        for field in ("package_manifest_sha256", "zip_sha256"):
            _require(bool(HASH_RE.fullmatch(record[field])), f"Invalid {field}")
        _require(_sha256(manifest) == record["package_manifest_sha256"],
                 "Package manifest hash mismatch")
        _require(_sha256(archive) == record["zip_sha256"], "ZIP hash mismatch")
        identity = _validate_identity(_verify_package(root, manifest, archive))
        _require(record["checkout_commit"] == identity["checkout_commit"],
                 "Checkout commit mismatch")
        _require(record["experiment_id"] == identity["manifest_sha256"],
                 "Experiment identity mismatch")
        _check_existing_records(record, existing_decision_paths)
    except Exception as exc:
        errors.append(str(exc))
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": not errors,
        "errors": sorted(set(errors)),
        "decision_record_sha256": (
            anchor.get("decision_record_sha256")
            if "anchor" in locals() else None
        ),
    }
