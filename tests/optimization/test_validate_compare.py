import csv
import json
from lawless.optimization.validate_compare import (
    validate_comparison, METHODS,
)




def write_json(root, name, value):
    (root / name).write_text(json.dumps(value), encoding="utf-8")




def write_csv(root, name, rows):
    with (root / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def make_evidence(root):
    seeds = [11, 23]
    manifest = {
        "methods": list(METHODS), "seeds": seeds, "seed": 7,
        "generations": 2, "population": 4, "sigma": 0.1,
        "candidate_budget": 8, "episode_budget": 16,
    }
    summaries, metrics = [], []

    for method in METHODS:
        summaries.append({
            "method": method, "candidate_evaluations": 8,
            "episodes": 16, "candidate_budget": 8,
            "episode_budget": 16, "runtime_seconds": 1.0,
            "best_fitness": 0.75, "best_vector": [0.1, 0.2],
            "seed_schedule": seeds, "seed": 7,
            "generations": 2, "population": 4, "sigma": 0.1,
        })

        metrics.extend([
            {"method": method, "generation": 1,
             "candidate_evaluations": 4, "episodes_total": 8,
             "best_fitness": 0.5, "mean_fitness": 0.25,
             "best_so_far": 0.5},
            {"method": method, "generation": 2,
             "candidate_evaluations": 8, "episodes_total": 16,
             "best_fitness": 0.75, "mean_fitness": 0.4,
             "best_so_far": 0.75},
        ])

    write_json(root, "manifest.json", manifest)
    write_json(root, "summary.json", summaries)
    write_json(root, "metrics.json", metrics)
    write_csv(root, "summary.csv", summaries)
    write_csv(root, "metrics.csv", metrics)
    return manifest, summaries, metrics




def rewrite(root, name, value):
    write_json(root, name, value)

def test_valid_evidence_passes(tmp_path):
    make_evidence(tmp_path)
    report = validate_comparison(tmp_path)
    assert report["passed"] is True
    assert report["failures"] == []
    assert (tmp_path / "validation_report.json").is_file()




def test_missing_method_fails(tmp_path):
    _, summaries, _ = make_evidence(tmp_path)
    rewrite(tmp_path, "summary.json", summaries[:-1])
    report = validate_comparison(tmp_path)
    assert not report["passed"]

def test_duplicate_method_fails(tmp_path):
    _, summaries, _ = make_evidence(tmp_path)
    summaries[-1] = dict(summaries[0])
    rewrite(tmp_path, "summary.json", summaries)
    report = validate_comparison(tmp_path)
    assert not report["passed"]




def test_seed_manifest_mismatch_fails(tmp_path):
    _, summaries, _ = make_evidence(tmp_path)
    summaries[0]["seed_schedule"] = [11, 99]
    rewrite(tmp_path, "summary.json", summaries)
    report = validate_comparison(tmp_path)
    assert not report["passed"]

def test_budget_mismatch_fails(tmp_path):
    _, summaries, _ = make_evidence(tmp_path)
    summaries[0]["episodes"] = 15
    rewrite(tmp_path, "summary.json", summaries)
    report = validate_comparison(tmp_path)
    assert not report["passed"]




def test_missing_generation_fails(tmp_path):
    _, _, metrics = make_evidence(tmp_path)
    metrics = [r for r in metrics
               if not (r["method"] == "GA" and r["generation"] == 2)]
    rewrite(tmp_path, "metrics.json", metrics)
    report = validate_comparison(tmp_path)
    assert not report["passed"]

def test_non_monotonic_counts_fail(tmp_path):
    _, _, metrics = make_evidence(tmp_path)
    row = next(r for r in metrics
               if r["method"] == "GA" and r["generation"] == 2)
    row["candidate_evaluations"] = 3
    rewrite(tmp_path, "metrics.json", metrics)
    report = validate_comparison(tmp_path)
    assert not report["passed"]

def test_nonfinite_values_fail(tmp_path):
    _, _, metrics = make_evidence(tmp_path)
    metrics[0]["mean_fitness"] = float("nan")
    rewrite(tmp_path, "metrics.json", metrics)
    report = validate_comparison(tmp_path)
    assert not report["passed"]




def test_best_so_far_regression_fails(tmp_path):
    _, _, metrics = make_evidence(tmp_path)
    metrics[1]["best_so_far"] = 0.1
    rewrite(tmp_path, "metrics.json", metrics)
    report = validate_comparison(tmp_path)
    assert not report["passed"]

def test_json_csv_disagreement_fails(tmp_path):
    _, _, metrics = make_evidence(tmp_path)
    metrics[0]["mean_fitness"] = 0.9
    rewrite(tmp_path, "metrics.json", metrics)
    report = validate_comparison(tmp_path)
    assert not report["passed"]
    assert any(f["code"] == "CSV_VALUE" for f in report["failures"])
