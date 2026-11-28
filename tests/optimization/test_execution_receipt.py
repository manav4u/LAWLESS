import json
import subprocess
from datetime import datetime, timedelta, timezone


import pytest


from lawless.optimization import execution_receipt as receipt


START = datetime(2026, 11, 28, 10, 0, tzinfo=timezone.utc)
END = START + timedelta(minutes=5)

@pytest.fixture
def setup_receipt(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    root.mkdir()
    evidence.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True,
                   capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"],
                   cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"],
                   cwd=root, check=True)

    (root / "tracked.txt").write_text("tracked\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"],
                   cwd=root, check=True, capture_output=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=root, text=True
    ).strip()

    manifest = {"methods": ["ga", "es", "cmaes"], "seed": 17}
    (evidence / "manifest.json").write_text(json.dumps(manifest))
    identity = receipt._canonical_hash(manifest)
    authorization = {
        "authorized": True,
        "experiment_manifest_sha256": identity,
        "max_candidate_evaluations": 100,
        "checkout_commit": commit,
    }

    auth_path = evidence / "authorization.json"
    auth_path.write_text(json.dumps(authorization))
    (evidence / "result.json").write_text('{"fitness": 0.5}')
    (evidence / "c2.json").write_text('{"valid": true}')
    (evidence / "provenance.json").write_text("{}")
    monkeypatch.setattr(
        receipt, "_git_identity",
        lambda _: {"commit": commit, "branch": branch},
    )

    monkeypatch.setattr(
        "lawless.optimization.provenance.verify_provenance",
        lambda *a: {"valid": True, "failures": []},
    )


    def create(status="completed", paths=("result.json",), budget=50):
        return receipt.create_execution_receipt(
            repo_root=root, evidence_dir=evidence,
            authorization_path=auth_path, artifact_paths=paths,
            consumed_budget=budget, started_utc=START, ended_utc=END,
            terminal_status=status, receipt_path=evidence / "receipt.json",
        )

    def verify():
        return receipt.verify_execution_receipt(
            receipt_path=evidence / "receipt.json",
            repo_root=root, evidence_dir=evidence,
            authorization_path=auth_path,
            c2_report_path=evidence / "c2.json",
            provenance_path=evidence / "provenance.json",
            source_files=(),
        )


    return root, evidence, create, verify

def test_valid_completed_receipt_passes(setup_receipt):
    _, _, create, verify = setup_receipt
    record = create()
    assert record["terminal_status"] == "completed"
    assert record["consumed_candidate_evaluations"] == 50
    assert verify()["valid"] is True




@pytest.mark.parametrize("status", ["failed", "interrupted", "timed_out"])
def test_noncompleted_statuses_are_recorded(setup_receipt, status):
    _, _, create, _ = setup_receipt
    assert create(status=status)["terminal_status"] == status

def test_missing_artifact_is_rejected(setup_receipt):
    _, evidence, create, verify = setup_receipt
    create()
    (evidence / "result.json").unlink()
    assert not verify()["valid"]




def test_modified_artifact_is_rejected(setup_receipt):
    _, evidence, create, verify = setup_receipt
    create()
    (evidence / "result.json").write_text("tampered")
    assert not verify()["valid"]

def test_duplicate_artifact_path_rejected(setup_receipt):
    with pytest.raises(ValueError, match="Duplicate"):
        setup_receipt[2](paths=("result.json", "result.json"))




def test_path_traversal_rejected(setup_receipt):
    with pytest.raises(ValueError, match="Unsafe"):
        setup_receipt[2](paths=("../outside.json",))

def test_manifest_identity_mismatch_rejected(setup_receipt):
    _, evidence, create, verify = setup_receipt
    create()
    (evidence / "manifest.json").write_text('{"seed": 999}')
    report = verify()
    assert not report["valid"]
    assert any(
        f["code"] == "manifest_identity_mismatch"
        for f in report["failures"]
    )

def test_authorization_identity_mismatch_rejected(setup_receipt):
    _, evidence, create, verify = setup_receipt
    create()
    path = evidence / "authorization.json"
    approval = json.loads(path.read_text())
    approval["max_candidate_evaluations"] = 99
    path.write_text(json.dumps(approval))
    assert not verify()["valid"]




def test_budget_overrun_rejected(setup_receipt):
    with pytest.raises(ValueError, match="exceeds"):
        setup_receipt[2](budget=101)

def test_invalid_interval_rejected(setup_receipt):
    _, _, create, _ = setup_receipt
    with pytest.raises(ValueError, match="precedes"):
        create_time = START - timedelta(seconds=1)
        receipt.create_execution_receipt(
            repo_root=setup_receipt[0],
            evidence_dir=setup_receipt[1],
            authorization_path=setup_receipt[1] / "authorization.json",
            artifact_paths=("result.json",), consumed_budget=10,
            started_utc=START, ended_utc=create_time,
            terminal_status="completed",
            receipt_path=setup_receipt[1] / "bad.json",
        )

def test_missing_c2_and_failed_c3_rejected(setup_receipt, monkeypatch):
    _, evidence, create, verify = setup_receipt
    create()
    (evidence / "c2.json").write_text('{"valid": false}')
    assert not verify()["valid"]
    (evidence / "c2.json").write_text('{"valid": true}')
    monkeypatch.setattr(
        "lawless.optimization.provenance.verify_provenance",
        lambda *a: {"valid": False, "failures": [{"code": "hash"}]},
    )
    assert not verify()["valid"]

def test_malformed_receipt_rejected(setup_receipt):
    _, evidence, _, verify = setup_receipt
    (evidence / "receipt.json").write_text("{broken")
    report = verify()
    assert not report["valid"]
    assert report["failures"][0]["code"] == "receipt_unreadable"




def test_missing_c2_report_rejected(setup_receipt):
    _, evidence, create, verify = setup_receipt
    create()
    (evidence / "c2.json").unlink()
    assert not verify()["valid"]

def test_duration_mismatch_rejected(setup_receipt):
    _, evidence, create, verify = setup_receipt
    create()
    path = evidence / "receipt.json"
    record = json.loads(path.read_text())
    record["elapsed_seconds"] = 999
    path.write_text(json.dumps(record))
    report = verify()
    assert not report["valid"]
    assert any(
        f["code"] == "duration_mismatch" for f in report["failures"]
    )

def test_duplicate_artifact_reference_rejected(setup_receipt):
    _, evidence, create, verify = setup_receipt
    create()
    path = evidence / "receipt.json"
    record = json.loads(path.read_text())
    record["artifacts"].append(record["artifacts"][0])
    path.write_text(json.dumps(record))
    report = verify()
    assert not report["valid"]

def test_verification_is_deterministic(setup_receipt):
    _, _, create, verify = setup_receipt
    create()
    first = verify()
    second = verify()
    assert first == second


import pytest
from lawless.optimization.execution_receipt import _consumed_int
def test_zero_consumed_allowed_interrupted():
    assert _consumed_int(0, "Consumed", "interrupted") == 0

def test_zero_consumed_allowed_timed_out():
    assert _consumed_int(0, "Consumed", "timed_out") == 0
def test_zero_consumed_allowed_failed():
    assert _consumed_int(0, "Consumed", "failed") == 0

def test_zero_consumed_completed_rejected():
    with pytest.raises(ValueError):
        _consumed_int(0, "Consumed", "completed")

def test_consumed_bool_rejected():
    with pytest.raises(ValueError):
        _consumed_int(True, "Consumed", "failed")
def test_negative_consumed_rejected():
    with pytest.raises(ValueError):
        _consumed_int(-1, "Consumed", "timed_out")
