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
ARTIFACT_FILES = (
    "manifest.json",
    "summary.json",
    "metrics.json",
    "summary.csv",
    "metrics.csv",
)
SOURCE_FILES = (
    "lawless/optimization/ga.py",
    "lawless/optimization/es.py",
    "lawless/optimization/cmaes.py",
    "lawless/optimization/evaluator.py",
    "lawless/optimization/compare.py",
    "lawless/optimization/validate_compare.py",
)

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _safe_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes root: {relative}")
    return resolved




def _git(root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=root, check=True,
            capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def _git_metadata(root: Path) -> dict[str, Any]:
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    status = _git(root, "status", "--porcelain")
    return {
        "commit": commit,
        "branch": branch,
        "working_tree": (
            "unknown" if status is None
            else "dirty" if status else "clean"
        ),
    }




def _torch_version() -> str | None:
    try:
        import torch
        return str(torch.__version__)
    except ImportError:
        return None

def _hash_files(
    root: Path, relatives: Iterable[str],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in relatives:
        path = _safe_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(relative)
        hashes[relative] = _sha256(path)
    return hashes




def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)

def create_provenance(
    repo_root: str | Path,
    c1_output_dir: str | Path,
    provenance_path: str | Path | None = None,
    source_files: Iterable[str] = SOURCE_FILES,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    evidence = Path(c1_output_dir).resolve()
    target = (
        Path(provenance_path).resolve()
        if provenance_path is not None
        else evidence / "provenance.json"
    )
    manifest_path = _safe_path(evidence, "manifest.json")
    manifest = _load_json(manifest_path)

    artifacts = _hash_files(evidence, ARTIFACT_FILES)
    sources = _hash_files(root, tuple(source_files))
    manifest_identity = _canonical_hash(manifest)
    git = _git_metadata(root)


    record = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "experiment": {
            "manifest_sha256": manifest_identity,
            "manifest": manifest,
        },

        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "pytorch_version": _torch_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "execution_device": "cpu",
        },
        "artifacts": artifacts,
        "sources": sources,
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(
            record, stream, indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )
        stream.write("\n")
    return record




def _failure(
    failures: list[dict[str, str]], code: str, detail: str,
) -> None:
    failures.append({"code": code, "detail": detail})

def verify_provenance(
    provenance_path: str | Path,
    repo_root: str | Path,
    c1_output_dir: str | Path,
    source_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []
    provenance_file = Path(provenance_path).resolve()
    root = Path(repo_root).resolve()
    evidence = Path(c1_output_dir).resolve()


    try:
        record = _load_json(provenance_file)
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "failures": [
            {"code": "provenance_unreadable", "detail": str(exc)}
        ]}

    if not isinstance(record, dict):
        return {"valid": False, "failures": [{
            "code": "invalid_schema", "detail": "Root must be an object"
        }]}


    if record.get("schema_version") != SCHEMA_VERSION:
        _failure(failures, "schema_version", "Unsupported schema version")


    required = {
        "created_utc", "git", "experiment", "runtime",
        "artifacts", "sources", "schema_version",
    }
    for key in sorted(required - record.keys()):
        _failure(failures, "missing_field", key)

    experiment = record.get("experiment")
    if not isinstance(experiment, dict):
        experiment = {}
        _failure(failures, "invalid_experiment", "Expected object")


    try:
        manifest = _load_json(_safe_path(evidence, "manifest.json"))
        actual_identity = _canonical_hash(manifest)
        if actual_identity != experiment.get("manifest_sha256"):
            _failure(failures, "manifest_mismatch",
                     "C1 manifest identity differs")
        if manifest != experiment.get("manifest"):
            _failure(failures, "manifest_content_mismatch",
                     "Recorded manifest content differs")

    except (OSError, ValueError, json.JSONDecodeError) as exc:
        _failure(failures, "manifest_unreadable", str(exc))


    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        _failure(failures, "invalid_artifacts", "Expected object")


    if set(artifacts) != set(ARTIFACT_FILES):
        _failure(failures, "artifact_set_mismatch",
                 "Artifact list differs from required C1 evidence set")

    for relative in ARTIFACT_FILES:
        try:
            actual = _sha256(_safe_path(evidence, relative))
            if artifacts.get(relative) != actual:
                _failure(failures, "artifact_hash_mismatch", relative)
        except (OSError, ValueError) as exc:
            _failure(failures, "artifact_missing", f"{relative}: {exc}")


    sources = record.get("sources")
    if not isinstance(sources, dict):
        sources = {}
        _failure(failures, "invalid_sources", "Expected object")

    expected_sources = (
        tuple(source_files) if source_files is not None
        else tuple(sources)
    )
    if set(sources) != set(expected_sources):
        _failure(failures, "source_set_mismatch",
                 "Recorded source list differs from expected source list")


    for relative in expected_sources:
        try:
            actual = _sha256(_safe_path(root, relative))
            if sources.get(relative) != actual:
                _failure(failures, "source_hash_mismatch", relative)

        except (OSError, ValueError) as exc:
            _failure(failures, "source_missing", f"{relative}: {exc}")


    git_record = record.get("git")
    if not isinstance(git_record, dict):
        _failure(failures, "invalid_git", "Expected object")
    else:
        for key in ("commit", "branch", "working_tree"):
            if key not in git_record:
                _failure(failures, "missing_git_field", key)

    runtime = record.get("runtime")
    if not isinstance(runtime, dict):
        _failure(failures, "invalid_runtime", "Expected object")
    elif runtime.get("execution_device") != "cpu":
        _failure(failures, "non_cpu_execution",
                 "Provenance does not declare CPU execution")


    return {
        "valid": not failures,
        "failures": failures,
        "checked_artifacts": len(ARTIFACT_FILES),
        "checked_sources": len(expected_sources),
    }

def write_validation_report(
    report: dict[str, Any], report_path: str | Path,
) -> None:
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(
            report, stream, indent=2, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )
        stream.write("\n")

def main() -> int:
    import argparse


    parser = argparse.ArgumentParser(description="C3 provenance tools")
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--repo", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--provenance")
    parser.add_argument("--report")
    parser.add_argument("--source", action="append", dest="sources")
    args = parser.parse_args()
    sources = tuple(args.sources) if args.sources else SOURCE_FILES

    path = args.provenance or str(
        Path(args.evidence) / "provenance.json"
    )
    if args.mode == "create":
        create_provenance(args.repo, args.evidence, path, sources)
        return 0


    report = verify_provenance(path, args.repo, args.evidence, sources)
    if args.report:
        write_validation_report(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1




if __name__ == "__main__":
    raise SystemExit(main())
