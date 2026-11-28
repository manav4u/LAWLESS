import hashlib
import json
import zipfile


import pytest


from lawless.optimization import acceptance as mod




def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path

@pytest.fixture
def package(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir()
    manifest = write_json(root / "package.json", {
        "schema_version": 1, "manifest_sha256": "x" * 64,
        "inventory": {},
    })
    archive = root / "package.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("package_manifest.json", manifest.read_bytes())

    identity = {
        "manifest_sha256": "a" * 64,
        "checkout_commit": "b" * 40,
        "checkout_branch": "controllers/c6-evidence-package",
        "terminal_status": "completed",
        "authorized_candidate_evaluations": 12,
        "consumed_candidate_evaluations": 10,
    }
    monkeypatch.setattr(
        mod, "_verify_package", lambda *args: dict(identity)
    )
    return root, manifest, archive, identity

def create(package, decision="accepted", **overrides):
    root, manifest, archive, _ = package
    args = dict(
        decision=decision, decision_maker="reviewer-1",
        timestamp_utc="2026-11-28T10:00:00Z",
        rationale="Synthetic acceptance test",
    )
    args.update(overrides)
    return mod.create_acceptance_decision(
        root, manifest, archive, root.parent / "decision.json",
        root.parent / "report.json", **args,
    )

@pytest.mark.parametrize("decision", [
    "accepted", "rejected", "inconclusive",
])
def test_all_explicit_decisions(package, decision):
    report = create(package, decision)
    assert report["valid"] is True




@pytest.mark.parametrize("decision", ["yes", "pending", "", None])
def test_invalid_decisions_rejected(package, decision):
    with pytest.raises(ValueError):
        create(package, decision)

@pytest.mark.parametrize("field,value", [
    ("decision_maker", ""), ("rationale", ""),
    ("timestamp_utc", "not-a-date"),
    ("timestamp_utc", "2026-11-28T10:00:00"),
    ("timestamp_utc", "2026-11-28T10:00:00+05:00"),
])
def test_invalid_metadata_rejected(package, field, value):
    with pytest.raises(ValueError):
        create(package, **{field: value})

@pytest.mark.parametrize("status", [
    "failed", "interrupted", "timed_out",
])
def test_noncompleted_execution_cannot_be_decided(package, status):
    package[3]["terminal_status"] = status
    with pytest.raises(ValueError, match="completed"):
        create(package)




def test_consumed_budget_overrun_rejected(package):
    package[3]["consumed_candidate_evaluations"] = 13
    with pytest.raises(ValueError, match="budget"):
        create(package)

def test_zip_tampering_rejected(package):
    root, manifest, archive, _ = package
    create(package)
    with zipfile.ZipFile(archive, "a") as zf:
        zf.writestr("tampered.txt", "modified")
    report = mod.verify_acceptance_decision(
        root, manifest, archive, root.parent / "decision.json",
        verification_report_path=root.parent / "report.json",
    )
    assert not report["valid"]
    assert "ZIP hash mismatch" in report["errors"]

def test_duplicate_and_conflicting_records_rejected(package):
    root, manifest, archive, _ = package
    create(package)
    existing = [root.parent / "decision.json"]
    with pytest.raises(ValueError, match="Duplicate"):
        create(package, existing_decision_paths=existing)
    with pytest.raises(ValueError, match="Conflicting"):
        create(package, "rejected", existing_decision_paths=existing)

def test_manifest_tampering_rejected(package):
    root, manifest, archive, _ = package
    create(package)
    manifest.write_text('{"modified": true}', encoding="utf-8")
    report = mod.verify_acceptance_decision(
        root, manifest, archive, root.parent / "decision.json",
        verification_report_path=root.parent / "report.json",
    )
    assert not report["valid"]
    assert "Package manifest hash mismatch" in report["errors"]

def test_checkout_commit_mismatch_rejected(package):
    root, manifest, archive, identity = package
    create(package)
    identity["checkout_commit"] = "c" * 40
    report = mod.verify_acceptance_decision(
        root, manifest, archive, root.parent / "decision.json",
        verification_report_path=root.parent / "report.json",
    )
    assert not report["valid"]
    assert "Checkout commit mismatch" in report["errors"]

def test_c6_verification_failure_rejected(package, monkeypatch):
    monkeypatch.setattr(mod, "_verify_package",
                        lambda *args: (_ for _ in ()).throw(
                            ValueError("C6 tampering")))
    with pytest.raises(ValueError, match="C6"):
        create(package)




def test_decision_record_tampering_rejected(package):
    root, manifest, archive, _ = package
    create(package)
    path = root.parent / "decision.json"
    record = json.loads(path.read_text())
    record["decision"] = "rejected"
    write_json(path, record)
    report = mod.verify_acceptance_decision(
        root, manifest, archive, path,
        verification_report_path=root.parent / "report.json",
    )
    assert not report["valid"]
    assert "Decision record integrity mismatch" in report["errors"]

def test_deterministic_report_and_evidence_preservation(package):
    root, manifest, archive, _ = package
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.iterdir()}
    create(package)
    args = (root, manifest, archive, root.parent / "decision.json")
    first = mod.verify_acceptance_decision(
        *args, verification_report_path=root.parent / "report.json")
    second = mod.verify_acceptance_decision(
        *args, verification_report_path=root.parent / "report.json")
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in root.iterdir()}
    assert first == second
    assert before == after
