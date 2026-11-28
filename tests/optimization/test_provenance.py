import json


import pytest


from lawless.optimization import provenance




@pytest.fixture
def evidence_tree(tmp_path):
    root = tmp_path / "repo"
    evidence = tmp_path / "evidence"
    root.mkdir()
    evidence.mkdir()
    (root / "sample_source.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )

    manifest = {"methods": ["ga", "es", "cmaes"], "seed": 17}
    files = {
        "manifest.json": json.dumps(manifest),
        "summary.json": '{"ok": true}',
        "metrics.json": '{"generation": [0]}',
        "summary.csv": "method,fitness\nga,0.5\n",
        "metrics.csv": "method,generation\n ga,0\n",
    }
    for name, content in files.items():
        (evidence / name).write_text(content, encoding="utf-8")
    return root, evidence

@pytest.fixture
def make_record(evidence_tree, monkeypatch):
    root, evidence = evidence_tree
    monkeypatch.setattr(
        provenance, "_git_metadata",
        lambda _: {
            "commit": "a" * 40,
            "branch": "controllers/c3-provenance",
            "working_tree": "clean",
        },
    )

    def create():
        return provenance.create_provenance(
            root, evidence,
            source_files=("sample_source.py",),
        )


    return root, evidence, create

def _verify(root, evidence):
    return provenance.verify_provenance(
        evidence / "provenance.json",
        root,
        evidence,
        source_files=("sample_source.py",),
    )




def test_valid_provenance_passes(make_record):
    root, evidence, create = make_record
    create()
    report = _verify(root, evidence)
    assert report["valid"] is True
    assert report["failures"] == []

def test_modified_artifact_is_rejected(make_record):
    root, evidence, create = make_record
    create()
    (evidence / "summary.csv").write_text("tampered\n")
    report = _verify(root, evidence)
    assert not report["valid"]
    assert any(
        item["code"] == "artifact_hash_mismatch"
        for item in report["failures"]
    )

def test_modified_source_is_rejected(make_record):
    root, evidence, create = make_record
    create()
    (root / "sample_source.py").write_text("VALUE = 2\n")
    report = _verify(root, evidence)
    assert not report["valid"]
    assert any(
        item["code"] == "source_hash_mismatch"
        for item in report["failures"]
    )

def test_missing_artifact_is_rejected(make_record):
    root, evidence, create = make_record
    create()
    (evidence / "metrics.csv").unlink()
    report = _verify(root, evidence)
    assert not report["valid"]
    assert any(
        item["code"] == "artifact_missing"
        for item in report["failures"]
    )

def test_missing_source_is_rejected(make_record):
    root, evidence, create = make_record
    create()
    (root / "sample_source.py").unlink()
    report = _verify(root, evidence)
    assert not report["valid"]
    assert any(
        item["code"] == "source_missing"
        for item in report["failures"]
    )

def test_manifest_mismatch_is_rejected(make_record):
    root, evidence, create = make_record
    create()
    (evidence / "manifest.json").write_text(
        '{"methods": ["ga"], "seed": 99}', encoding="utf-8"
    )
    report = _verify(root, evidence)
    assert not report["valid"]
    assert any(
        item["code"] == "manifest_mismatch"
        for item in report["failures"]
    )

def test_dirty_checkout_is_recorded(make_record, monkeypatch):
    root, evidence, create = make_record
    monkeypatch.setattr(
        provenance, "_git_metadata",
        lambda _: {
            "commit": "b" * 40,
            "branch": "controllers/c3-provenance",
            "working_tree": "dirty",
        },
    )
    record = create()
    assert record["git"]["working_tree"] == "dirty"

def test_verification_is_deterministic(make_record):
    root, evidence, create = make_record
    create()
    first = _verify(root, evidence)
    second = _verify(root, evidence)
    assert first == second




def test_malformed_provenance_is_rejected(make_record):
    root, evidence, _ = make_record
    (evidence / "provenance.json").write_text("{broken")
    report = _verify(root, evidence)
    assert not report["valid"]
    assert report["failures"][0]["code"] == "provenance_unreadable"
