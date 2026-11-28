import hashlib
import json
import zipfile
from pathlib import Path


import pytest


from lawless.optimization.evidence_package import (
    REQUIRED, _json_bytes, create_evidence_package,
    verify_evidence_package,
)




def sha(data):
    return hashlib.sha256(data).hexdigest()




def write_json(path, value):
    path.write_bytes(_json_bytes(value) + b"\n")
    return path

@pytest.fixture
def evidence(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    c1 = {"schema_version": 1, "seeds": [11, 12], "methods": ["ga", "es", "cmaes"]}
    identity = sha(_json_bytes(c1))
    auth = {"schema_version": 1, "authorized": True,
            "experiment_manifest_sha256": identity,
            "checkout_commit": "abc123", "max_candidate_evaluations": 30}
    auth_path = write_json(root / "authorization.json", auth)
    docs = {
        "c1_manifest": c1,
        "c1_summary": {"results": []},
        "c1_metrics": {"generations": []},
        "c2_report": {"valid": True},

        "c3_provenance": {
            "schema_version": 1,
            "experiment": {"manifest_sha256": identity},
            "git": {"commit": "abc123", "branch": "controllers/test"},
            "verified": True,
        },
        "c4_authorization": auth,
        "c4_gate_decision": {
            "authorized": True, "checkout_commit": "abc123",
            "experiment_manifest_sha256": identity,
            "authorized_candidate_evaluations": 30,
        },
        "c5_receipt": {
            "schema_version": 1, "experiment_manifest_sha256": identity,
            "authorization_sha256": sha(auth_path.read_bytes()),
            "checkout_commit": "abc123", "checkout_branch": "controllers/test",

            "authorized_candidate_evaluations": 30,
            "consumed_candidate_evaluations": 24,
            "terminal_status": "completed",
        },
        "c5_verification_report": {"valid": True, "terminal_status": "completed"},
    }
    artifacts = {}
    for key in REQUIRED:
        if key == "c1_manifest":
            value = docs[key]
        elif key in docs:
            value = docs[key]
        else:
            value = "generation,fitness\n0,0.5\n"
        path = root / f"{key}.{'json' if isinstance(value, dict) else 'csv'}"
        artifacts[key] = write_json(path, value) if isinstance(value, dict) else path
        if not isinstance(value, dict):
            path.write_text(value, encoding="utf-8")
    return root, artifacts

def test_complete_package_passes_and_zip_matches_inventory(tmp_path, evidence):
    root, artifacts = evidence
    output = create_evidence_package(root, artifacts, tmp_path / "package")
    assert output["verification"]["valid"] is True
    manifest = json.loads(Path(output["package_manifest"]).read_text())
    assert set(manifest["inventory"]) == set(REQUIRED)
    with zipfile.ZipFile(output["zip_path"]) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names))
        assert "package_manifest.json" in names
        assert "package_verification_report.json" in names
        for item in manifest["inventory"].values():
            assert sha(archive.read(item["path"])) == item["sha256"]

def test_missing_required_artifact_rejected(tmp_path, evidence):
    root, artifacts = evidence
    artifacts.pop("c1_metrics")
    with pytest.raises(ValueError):
        create_evidence_package(root, artifacts, tmp_path / "out")




def test_duplicate_source_path_rejected(tmp_path, evidence):
    root, artifacts = evidence
    artifacts["c1_metrics"] = artifacts["c1_summary"]
    with pytest.raises(ValueError):
        create_evidence_package(root, artifacts, tmp_path / "out")




def test_path_traversal_rejected():
    from lawless.optimization.evidence_package import _safe_relative
    with pytest.raises(ValueError):
        _safe_relative("../outside.json")

@pytest.mark.parametrize("stage,field,value", [
    ("c3_provenance", "experiment", {"manifest_sha256": "wrong"}),
    ("c4_authorization", "checkout_commit", "different"),
    ("c4_authorization", "max_candidate_evaluations", 20),
    ("c5_receipt", "consumed_candidate_evaluations", 31),
    ("c5_receipt", "terminal_status", "unknown"),
])
def test_cross_stage_inconsistency_rejected(tmp_path, evidence, stage, field, value):
    root, artifacts = evidence
    path = Path(artifacts[stage])
    doc = json.loads(path.read_text())
    doc[field] = value
    write_json(path, doc)
    with pytest.raises(ValueError):
        create_evidence_package(root, artifacts, tmp_path / "out")

def test_failed_stage_reports_rejected(tmp_path, evidence):
    root, artifacts = evidence
    write_json(artifacts["c2_report"], {"valid": False})
    with pytest.raises(ValueError):
        create_evidence_package(root, artifacts, tmp_path / "out")




def test_unexpected_schema_version_rejected(tmp_path, evidence):
    root, artifacts = evidence
    doc = json.loads(Path(artifacts["c5_receipt"]).read_text())
    doc["schema_version"] = 99
    write_json(artifacts["c5_receipt"], doc)
    with pytest.raises(ValueError):
        create_evidence_package(root, artifacts, tmp_path / "out")

def test_modified_inventory_file_detected(tmp_path, evidence):
    root, artifacts = evidence
    out = create_evidence_package(root, artifacts, tmp_path / "out")
    Path(artifacts["c1_summary"]).write_text('{"tampered":true}')
    report = verify_evidence_package(root, out["package_manifest"], out["zip_path"])
    assert report["valid"] is False




@pytest.mark.parametrize("key", ["c3_provenance", "c5_verification_report"])
def test_failed_provenance_or_receipt_audit_rejected(tmp_path, evidence, key):
    root, artifacts = evidence
    write_json(artifacts[key], {"valid": False})
    with pytest.raises(ValueError):
        create_evidence_package(root, artifacts, tmp_path / "out")




def test_creation_preserves_original_evidence(tmp_path, evidence):
    root, artifacts = evidence
    before = {key: Path(path).read_bytes() for key, path in artifacts.items()}
    create_evidence_package(root, artifacts, tmp_path / "out")
    assert before == {key: Path(path).read_bytes() for key, path in artifacts.items()}
