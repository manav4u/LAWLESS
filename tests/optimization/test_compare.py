import json
import torch
from lawless.optimization.compare import (
    compare_optimizers, write_comparison, METHODS,
)




SEEDS = (11, 23, 37)




class FakeEvaluator:
    def __init__(self, seeds):
        self.seeds = tuple(seeds)


    def evaluate_many(self, candidates):
        return [-float(v.square().sum()) for v in candidates]

def factory_log(log):
    def factory(seeds):
        log.append(tuple(seeds))
        return FakeEvaluator(seeds)
    return factory




def run(log):
    return compare_optimizers(
        torch.zeros(4), factory_log(log), SEEDS,
        generations=3, population=6, sigma=0.1, seed=9,
    )

def test_seed_schedule_parity():
    log = []
    run(log)
    assert log == [SEEDS, SEEDS, SEEDS]




def test_equal_candidate_episode_budgets():
    result = run([])
    summaries = result["summaries"]
    assert [s["method"] for s in summaries] == list(METHODS)
    assert all(s["candidate_evaluations"] == 18 for s in summaries)
    assert all(s["episodes"] == 54 for s in summaries)
    assert all(s["candidate_budget"] == 18 for s in summaries)
    assert all(s["episode_budget"] == 54 for s in summaries)

def test_generation_metrics_recorded():
    result = run([])
    records = result["records"]
    assert len(records) == 9
    for method in METHODS:
        rows = [r for r in records if r["method"] == method]
        assert [r["generation"] for r in rows] == [1, 2, 3]
        assert [r["candidate_evaluations"] for r in rows] == [6, 12, 18]
        assert [r["episodes_total"] for r in rows] == [18, 36, 54]

def test_metric_fields_and_finite_values():
    result = run([])
    for row in result["records"]:
        assert "best_fitness" in row
        assert "mean_fitness" in row
        assert "best_so_far" in row
        assert all(torch.isfinite(torch.tensor(row[k]))
                   for k in ("best_fitness", "mean_fitness",
                             "best_so_far"))

def test_reproducibility():
    a, b = run([]), run([])
    assert a["manifest"] == b["manifest"]
    assert a["records"] == b["records"]
    for x, y in zip(a["summaries"], b["summaries"]):
        assert x["best_fitness"] == y["best_fitness"]
        assert x["best_vector"] == y["best_vector"]

def test_json_and_csv_written(tmp_path):
    result = run([])
    write_comparison(result, tmp_path)
    for name in ("manifest.json", "summary.json", "metrics.json",
                 "summary.csv", "metrics.csv"):
        assert (tmp_path / name).is_file()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["methods"] == list(METHODS)
    assert manifest["seeds"] == list(SEEDS)

def test_summary_contains_best_and_runtime():
    result = run([])
    for summary in result["summaries"]:
        assert len(summary["best_vector"]) == 4
        assert summary["best_fitness"] <= 0
        assert summary["runtime_seconds"] >= 0
        assert summary["seed_schedule"] == list(SEEDS)
