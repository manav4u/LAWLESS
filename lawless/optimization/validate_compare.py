import ast
import csv
import json
import math
from pathlib import Path


METHODS = ("GA", "OpenAI-ES", "CMA-ES")




def _read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)




def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

def _fail(failures, code, message, method=None):
    item = {"code": code, "message": message}
    if method is not None:
        item["method"] = method
    failures.append(item)




def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False

def _csv_value(key, value):
    if key in ("best_vector", "seed_schedule"):
        return ast.literal_eval(value)
    if key in ("method",):
        return value
    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return value

def _same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-12,
                            abs_tol=1e-12)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(
            _same(x, y) for x, y in zip(a, b))
    return a == b

def _check_csv_agreement(json_rows, csv_rows, label, failures):
    if len(json_rows) != len(csv_rows):
        _fail(failures, "CSV_ROW_COUNT",
              f"{label}: JSON/CSV row counts differ")
        return
    for index, (jrow, crow) in enumerate(zip(json_rows, csv_rows)):
        if set(jrow) != set(crow):
            _fail(failures, "CSV_COLUMNS",
                  f"{label}: columns differ at row {index}")
            continue

        for key, expected in jrow.items():
            try:
                actual = _csv_value(key, crow[key])
                if not _same(expected, actual):
                    _fail(failures, "CSV_VALUE",
                          f"{label}: {key} differs at row {index}")
            except (ValueError, SyntaxError, TypeError, KeyError):
                _fail(failures, "CSV_PARSE",
                      f"{label}: cannot parse {key} at row {index}")

def validate_comparison(output_dir, report_path=None):
    root = Path(output_dir)
    failures = []
    report_path = Path(report_path) if report_path else (
        root / "validation_report.json")
    required = ("manifest.json", "summary.json", "metrics.json",
                "summary.csv", "metrics.csv")
    missing = [name for name in required if not (root / name).is_file()]
    for name in missing:
        _fail(failures, "MISSING_FILE", f"Missing artifact: {name}")

    if missing:
        report = {"passed": False, "failures": failures}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    try:
        manifest = _read_json(root / "manifest.json")
        summaries = _read_json(root / "summary.json")
        metrics = _read_json(root / "metrics.json")
        summary_csv = _read_csv(root / "summary.csv")
        metrics_csv = _read_csv(root / "metrics.csv")
    except (OSError, ValueError, TypeError) as exc:

        _fail(failures, "READ_ERROR", str(exc))
        report = {"passed": False, "failures": failures}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


    required_manifest = (
        "methods", "seeds", "generations", "population",
        "candidate_budget", "episode_budget", "sigma", "seed",
    )

    for key in required_manifest:
        if key not in manifest:
            _fail(failures, "MANIFEST_FIELD",
                  f"Manifest missing {key}")
    if any(key not in manifest for key in required_manifest):
        manifest_ok = False
    else:
        manifest_ok = True
    if not manifest_ok:
        report = {"passed": False, "failures": failures}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    seeds = manifest["seeds"]
    generations = manifest["generations"]
    population = manifest["population"]
    candidate_budget = manifest["candidate_budget"]
    episode_budget = manifest["episode_budget"]
    valid = (isinstance(seeds, list) and len(seeds) > 0
             and isinstance(generations, int) and generations >= 1
             and isinstance(population, int) and population >= 2)
    if not valid:
        _fail(failures, "MANIFEST_INVALID",
              "Invalid seeds, generations, or population")
        report = {"passed": False, "failures": failures}
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    expected_candidates = generations * population
    expected_episodes = expected_candidates * len(seeds)
    if candidate_budget != expected_candidates:
        _fail(failures, "MANIFEST_BUDGET",
              "Manifest candidate budget is inconsistent")
    if episode_budget != expected_episodes:
        _fail(failures, "MANIFEST_BUDGET",
              "Manifest episode budget is inconsistent")
    if manifest["methods"] != list(METHODS):
        _fail(failures, "MANIFEST_METHODS",
              "Manifest method list must contain the three methods in order")

    if not isinstance(summaries, list) or not isinstance(metrics, list):
        _fail(failures, "DATA_FORMAT",
              "Summary and metrics JSON must contain arrays")
        summaries = summaries if isinstance(summaries, list) else []
        metrics = metrics if isinstance(metrics, list) else []


    summary_methods = [row.get("method") for row in summaries
                       if isinstance(row, dict)]
    metric_methods = [row.get("method") for row in metrics
                      if isinstance(row, dict)]
    for method in METHODS:
        if summary_methods.count(method) != 1:
            _fail(failures, "METHOD_COUNT",
                  "Summary must contain method exactly once", method)

        if metric_methods.count(method) != generations:
            _fail(failures, "METRIC_METHOD_COUNT",
                  "Metrics must contain one row per generation", method)
    if set(summary_methods) != set(METHODS):
        _fail(failures, "METHOD_SET",
              "Summary must contain exactly the three methods")
    if set(metric_methods) != set(METHODS):
        _fail(failures, "METRIC_METHOD_SET",
              "Metrics must contain exactly the three methods")

    for method in METHODS:
        rows = [r for r in summaries if isinstance(r, dict)
                and r.get("method") == method]
        if len(rows) != 1:
            continue
        row = rows[0]
        if row.get("seed_schedule") != seeds or row.get("seed") != manifest["seed"]:
            _fail(failures, "SEED_MISMATCH",
                  "Summary seed configuration differs from manifest", method)
        expected = {
            "candidate_evaluations": candidate_budget,
            "episodes": episode_budget,
            "candidate_budget": candidate_budget,
            "episode_budget": episode_budget,
        }

        for key, value in expected.items():
            if row.get(key) != value:
                _fail(failures, "SUMMARY_BUDGET",
                      f"Summary {key} differs from manifest", method)
        vector = row.get("best_vector")
        if (not isinstance(vector, list) or not vector
                or not all(_finite(x) for x in vector)):
            _fail(failures, "NONFINITE_VECTOR",
                  "Best parameter vector missing or nonfinite", method)
        for key in ("best_fitness", "runtime_seconds"):
            if not _finite(row.get(key)):
                _fail(failures, "NONFINITE_SUMMARY",
                      f"Summary {key} missing or nonfinite", method)

    for method in METHODS:
        rows = [r for r in metrics if isinstance(r, dict)
                and r.get("method") == method]
        expected_generations = list(range(1, generations + 1))
        actual_generations = [r.get("generation") for r in rows]
        if actual_generations != expected_generations:
            _fail(failures, "GENERATION_SEQUENCE",
                  "Generation rows missing, duplicated, or out of order",
                  method)

        previous_candidates, previous_episodes = 0, 0
        previous_best = float("-inf")
        for index, row in enumerate(rows):
            expected_candidates = (index + 1) * population
            expected_episodes = expected_candidates * len(seeds)
            candidates = row.get("candidate_evaluations")
            episodes = row.get("episodes_total")
            if (not isinstance(candidates, int)
                    or not isinstance(episodes, int)
                    or candidates <= previous_candidates
                    or episodes <= previous_episodes):

                _fail(failures, "NON_MONOTONIC_COUNTS",
                      "Evaluation counters must increase each generation",
                      method)
            if candidates != expected_candidates or episodes != expected_episodes:
                _fail(failures, "GENERATION_BUDGET",
                      "Generation counters disagree with manifest", method)
            previous_candidates, previous_episodes = candidates, episodes
            for key in ("best_fitness", "mean_fitness", "best_so_far"):
                if not _finite(row.get(key)):

                    _fail(failures, "NONFINITE_METRIC",
                          f"Metric {key} missing or nonfinite", method)
            best = row.get("best_so_far")
            if _finite(best):
                if best < previous_best:
                    _fail(failures, "BEST_REGRESSION",
                          "best_so_far regressed", method)
                previous_best = max(previous_best, best)

    _check_csv_agreement(summaries, summary_csv, "summary", failures)
    _check_csv_agreement(metrics, metrics_csv, "metrics", failures)
    report = {
        "passed": not failures,
        "methods_checked": list(METHODS),
        "generations_expected": generations,
        "candidate_budget_expected": candidate_budget,
        "episode_budget_expected": episode_budget,
        "failures": failures,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Validate C1 optimizer comparison evidence")
    parser.add_argument("output_dir")
    parser.add_argument("--report")
    args = parser.parse_args()
    result = validate_comparison(args.output_dir, args.report)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)




if __name__ == "__main__":
    main()
