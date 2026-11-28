from __future__ import annotations
import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


SCHEMA_VERSION = 1
REQUIRED = (
    "c1_manifest", "c1_summary", "c1_metrics",
    "c1_summary_csv", "c1_metrics_csv", "c2_report",
    "c3_provenance", "c4_authorization", "c4_gate_decision",
    "c5_receipt", "c5_verification_report",
)
TERMINAL = {"completed", "failed", "interrupted", "timed_out"}

def _json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")




def _sha(data):
    return hashlib.sha256(data).hexdigest()




def _file_sha(path):
    return _sha(Path(path).read_bytes())




def _safe_relative(value):
    p = PurePosixPath(str(value).replace("\\", "/"))
    if p.is_absolute() or not p.parts or ".." in p.parts:
        raise ValueError(f"unsafe relative path: {value}")
    if p.as_posix() in {"", "."}:
        raise ValueError(f"invalid relative path: {value}")
    return p.as_posix()

def _read_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value




def _result(errors, **extra):
    return {"schema_version": SCHEMA_VERSION,
            "valid": not errors, "errors": sorted(set(errors)), **extra}




def _positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0

def _inventory(root, artifacts):
    if set(artifacts) != set(REQUIRED):
        missing = sorted(set(REQUIRED) - set(artifacts))
        extra = sorted(set(artifacts) - set(REQUIRED))
        raise ValueError(f"artifact keys mismatch; missing={missing}, extra={extra}")
    root = Path(root).resolve()
    entries, seen = {}, set()
    for key in REQUIRED:
        source = Path(artifacts[key]).resolve()
        try:
            relative = source.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"{key} is outside evidence root") from exc
        relative = _safe_relative(relative)
        if relative in seen or not source.is_file():
            raise ValueError(f"duplicate path or missing file: {relative}")
        seen.add(relative)
        entries[key] = {"path": relative, "sha256": _file_sha(source)}
    return entries

def _audit(root, entries):
    errors = []
    docs = {}
    for key in REQUIRED:
        path = Path(root) / entries[key]["path"]
        if _file_sha(path) != entries[key]["sha256"]:
            errors.append(f"hash mismatch: {key}")
        if key.endswith(("manifest", "report", "provenance",
                         "authorization", "decision", "receipt")):
            try:
                docs[key] = _read_json(path)
            except Exception as exc:
                errors.append(f"invalid JSON {key}: {exc}")
    if errors:
        return errors, {}
    c1 = docs["c1_manifest"]
    identity = _sha(_json_bytes(c1))

    c2, c3 = docs["c2_report"], docs["c3_provenance"]
    if c2.get("valid") is not True:
        errors.append("C2 validation report is not valid")
    if c3.get("schema_version") != 1:
        errors.append("unexpected C3 schema version")
    if c3.get("verified") is not True:
        errors.append("C3 provenance verification is not successful")
    if c3.get("experiment", {}).get("manifest_sha256") != identity:
        errors.append("C3 manifest identity mismatch")
    git = c3.get("git", {})
    commit, branch = git.get("commit"), git.get("branch")
    if not isinstance(commit, str) or not commit:
        errors.append("C3 checkout commit missing")
    if not isinstance(branch, str) or not branch:
        errors.append("C3 checkout branch missing")

    auth, gate = docs["c4_authorization"], docs["c4_gate_decision"]
    if auth.get("schema_version") != 1:
        errors.append("unexpected C4 authorization schema version")
    if auth.get("authorized") is not True:
        errors.append("C4 authorization is not authorized")
    if auth.get("experiment_manifest_sha256") != identity:
        errors.append("C4 manifest identity mismatch")
    if auth.get("checkout_commit") != commit:
        errors.append("C4 checkout commit mismatch")
    cap = auth.get("max_candidate_evaluations")
    if not _positive_int(cap):
        errors.append("invalid C4 candidate-evaluation budget")

    if gate.get("authorized") is not True:
        errors.append("C4 gate decision is not authorized")
    if gate.get("checkout_commit") != commit:
        errors.append("C4 gate checkout commit mismatch")
    if gate.get("experiment_manifest_sha256") != identity:
        errors.append("C4 gate manifest identity mismatch")
    if gate.get("authorized_candidate_evaluations", cap) != cap:
        errors.append("C4 gate budget mismatch")
    receipt = docs["c5_receipt"]
    if receipt.get("schema_version") != 1:
        errors.append("unexpected C5 receipt schema version")
    if receipt.get("experiment_manifest_sha256") != identity:
        errors.append("C5 manifest identity mismatch")

    if receipt.get("authorization_sha256") != entries["c4_authorization"]["sha256"]:
        errors.append("C5 authorization hash mismatch")
    if receipt.get("checkout_commit") != commit:
        errors.append("C5 checkout commit mismatch")
    if receipt.get("checkout_branch") != branch:
        errors.append("C5 checkout branch mismatch")
    authorized = receipt.get("authorized_candidate_evaluations")
    consumed = receipt.get("consumed_candidate_evaluations")
    if authorized != cap:
        errors.append("C5 authorized budget mismatch")
    if not isinstance(consumed, int) or isinstance(consumed, bool) or consumed < 0 or consumed > cap:
        errors.append("C5 consumed budget exceeds authorization")

    status = receipt.get("terminal_status")
    if status not in TERMINAL:
        errors.append("invalid C5 terminal status")
    c5v = docs["c5_verification_report"]
    if c5v.get("valid") is not True:
        errors.append("C5 verification report is not valid")
    if c5v.get("terminal_status", status) != status:
        errors.append("C5 verification terminal status mismatch")
    if status == "completed":
        if c2.get("valid") is not True:
            errors.append("completed receipt requires valid C2")
        if c3.get("verified") is False:
            errors.append("completed receipt has failed C3 verification")
    return errors, {"manifest_sha256": identity, "checkout_commit": commit,
                    "checkout_branch": branch, "terminal_status": status,
                    "authorized_candidate_evaluations": cap,
                    "consumed_candidate_evaluations": consumed}

def verify_evidence_package(evidence_root, package_manifest_path, zip_path=None):
    errors = []
    try:
        manifest = _read_json(package_manifest_path)
        if manifest.get("schema_version") != SCHEMA_VERSION:
            errors.append("unexpected package schema version")
        entries = manifest.get("inventory", {})
        if set(entries) != set(REQUIRED):
            errors.append("package inventory members mismatch")
        for key, item in entries.items():
            path = _safe_relative(item["path"])
            if _file_sha(Path(evidence_root) / path) != item["sha256"]:
                errors.append(f"inventory hash mismatch: {key}")
        audit_errors, identity = _audit(evidence_root, entries)
        errors.extend(audit_errors)

        if zip_path is not None:
            with zipfile.ZipFile(zip_path) as archive:
                names = archive.namelist()
                expected = [v["path"] for v in entries.values()]
                expected += ["package_manifest.json", "package_verification_report.json"]
                if len(names) != len(set(names)):
                    errors.append("duplicate ZIP member names")
                if sorted(names) != sorted(expected):
                    errors.append("ZIP member list mismatch")
                for key, item in entries.items():
                    if _sha(archive.read(item["path"])) != item["sha256"]:
                        errors.append(f"ZIP content hash mismatch: {key}")
        return _result(errors, identity=identity)
    except Exception as exc:
        return _result([f"package verification error: {type(exc).__name__}: {exc}"])

def create_evidence_package(
    evidence_root, artifacts, output_dir, zip_name="evidence_package.zip"
):
    root, out = Path(evidence_root).resolve(), Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    inventory = _inventory(root, artifacts)
    errors, identity = _audit(root, inventory)
    if errors:
        raise ValueError("evidence audit failed: " + "; ".join(errors))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": identity["manifest_sha256"],
        "inventory": inventory,
    }
    manifest_path = out / "package_manifest.json"
    report_path = out / "package_verification_report.json"
    zip_path = out / _safe_relative(zip_name)

    manifest_path.write_bytes(_json_bytes(manifest) + b"\n")
    report = verify_evidence_package(root, manifest_path)
    report_path.write_bytes(_json_bytes(report) + b"\n")
    if report["valid"] is not True:
        raise ValueError("package verification failed: " + "; ".join(report["errors"]))
    return _write_package_zip(
        root, out, inventory, manifest_path, report_path, zip_path
    )

def _write_package_zip(root, out, inventory, manifest_path, report_path, zip_path):
    if zip_path.resolve() in {manifest_path.resolve(), report_path.resolve()}:
        raise ValueError("ZIP path collides with package metadata")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in inventory.values():
            archive.write(root / item["path"], item["path"])
        archive.write(manifest_path, "package_manifest.json")
        archive.write(report_path, "package_verification_report.json")
    result = verify_evidence_package(root, manifest_path, zip_path)
    report_path.write_bytes(_json_bytes(result) + b"\n")

    final = verify_evidence_package(root, manifest_path, zip_path)
    if not final["valid"]:
        raise ValueError("final ZIP verification failed: " + "; ".join(final["errors"]))
    report_path.write_bytes(_json_bytes(final) + b"\n")
    return {"package_manifest": str(manifest_path), "verification_report": str(report_path),
            "zip_path": str(zip_path), "verification": final}

def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify a C6 evidence package")
    parser.add_argument("evidence_root")
    parser.add_argument("package_manifest")
    parser.add_argument("--zip")
    parser.add_argument("--report")
    args = parser.parse_args(argv)
    report = verify_evidence_package(args.evidence_root, args.package_manifest, args.zip)
    output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.report:
        Path(args.report).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report["valid"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
