import json
import subprocess
from datetime import datetime, timedelta, timezone


import pytest


from lawless.optimization import experiment_gate as gate




NOW = datetime(2026, 11, 28, 12, 0, tzinfo=timezone.utc)

@pytest.fixture
def setup_gate(tmp_path, monkeypatch):
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

    manifest = {"methods": ["ga", "es", "cmaes"], "seed": 17}
    (evidence / "manifest.json").write_text(json.dumps(manifest))
    identity = gate._canonical_hash(manifest)
    (evidence / "provenance.json").write_text("{}")
    (evidence / "c2.json").write_text('{"valid": true}')
    (evidence / "authorization.json").write_text("{}")
    monkeypatch.setattr(gate, "_verify_c3", lambda *a: (True, ""))

    def approval(**changes):
        value = {
            "schema_version": 1,
            "authorized": True,
            "issued_utc": (NOW - timedelta(minutes=5)).isoformat(),
            "expires_utc": (NOW + timedelta(hours=2)).isoformat(),
            "checkout_commit": commit,
            "experiment_manifest_sha256": identity,
            "max_candidate_evaluations": 100,
        }
        value.update(changes)
        (evidence / "authorization.json").write_text(json.dumps(value))

    def inspect(budget=50, **kwargs):
        options = {
            "repo_root": root,
            "evidence_dir": evidence,
            "provenance_path": evidence / "provenance.json",
            "c2_report_path": evidence / "c2.json",
            "authorization_path": evidence / "authorization.json",
            "requested_budget": budget,
            "source_files": (),
        }
        options.update(kwargs)
        options.setdefault("now", NOW)
        return gate.inspect_experiment(**options)


    approval()
    return root, evidence, approval, inspect, commit

def test_valid_authorization_allows_dry_run(setup_gate):
    _, _, _, inspect, _ = setup_gate
    result = inspect()
    assert result["authorized"] is True
    assert result["decision"] == "authorized"
    assert result["dry_run"] is True
    assert result["experiment_launched"] is False




def test_missing_authorization_is_rejected(setup_gate):
    _, evidence, _, inspect, _ = setup_gate
    (evidence / "authorization.json").unlink()
    assert inspect()["authorized"] is False

def test_invalid_c2_report_is_rejected(setup_gate):
    _, evidence, _, inspect, _ = setup_gate
    (evidence / "c2.json").write_text('{"valid": false}')
    result = inspect()
    assert not result["authorized"]
    assert result["failures"][0]["code"] == "c2_validation_failed"




def test_invalid_c3_provenance_is_rejected(setup_gate, monkeypatch):
    monkeypatch.setattr(
        gate, "_verify_c3", lambda *a: (False, "hash mismatch")
    )

    result = setup_gate[3]()
    assert not result["authorized"]
    assert result["failures"][0]["code"] == "c3_integrity_failed"




def test_changed_manifest_is_rejected(setup_gate):
    _, evidence, _, inspect, _ = setup_gate
    (evidence / "manifest.json").write_text('{"seed": 99}')
    result = inspect()
    assert not result["authorized"]
    assert result["failures"][0]["code"] == "experiment_identity_mismatch"

def test_changed_checkout_commit_is_rejected(setup_gate):
    _, _, approval, inspect, _ = setup_gate
    approval(checkout_commit="0" * 40)
    result = inspect()
    assert not result["authorized"]
    assert result["failures"][0]["code"] == "checkout_commit_mismatch"




def test_budget_over_limit_is_rejected(setup_gate):
    result = setup_gate[3](budget=101)
    assert not result["authorized"]
    assert result["failures"][0]["code"] == "budget_exceeded"

def test_malformed_authorization_is_rejected(setup_gate):
    _, evidence, _, inspect, _ = setup_gate
    (evidence / "authorization.json").write_text("{bad")
    result = inspect()
    assert not result["authorized"]
    assert result["failures"][0]["code"] == "authorization_unreadable"




def test_expired_authorization_is_rejected(setup_gate):
    _, _, approval, inspect, _ = setup_gate
    approval(expires_utc=(NOW - timedelta(seconds=1)).isoformat())

    result = inspect()
    assert not result["authorized"]
    assert result["failures"][0]["code"] == "authorization_expired"




def test_stale_authorization_is_rejected(setup_gate):
    _, _, approval, inspect, _ = setup_gate
    approval(issued_utc=(NOW - timedelta(days=2)).isoformat())
    result = inspect(max_age_seconds=3600)
    assert not result["authorized"]
    assert result["failures"][0]["code"] == "authorization_stale"

def test_naive_clock_is_rejected(setup_gate):
    _, _, _, inspect, _ = setup_gate
    result = inspect(now=datetime(2026, 11, 28, 12, 0))
    assert not result["authorized"]
    assert result["failures"][0]["code"] == "invalid_clock"




def test_repeated_inspection_is_deterministic(setup_gate):
    _, _, _, inspect, _ = setup_gate
    assert inspect() == inspect()
