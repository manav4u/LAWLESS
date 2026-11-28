import json, time, traceback
from datetime import datetime, timezone
from pathlib import Path
import torch
from lawless.optimization.compare import METHODS, build_optimizer
from lawless.optimization.evaluator import flatten_parameters
from lawless.optimization.pilot_adapter import PilotAdapter, PilotTimeout, PilotAccounting
from lawless.optimization.pilot_authorization import inspect_authorization, planned_manifest, planned_identity, write_json

SEARCH_SEEDS = (11, 23, 37)
HELDOUT_SEEDS = (101, 211, 307)
GENERATIONS, POPULATION, SIGMA, OPT_SEED = 10, 6, 0.1, 7
MAX_STEPS, WALL_SECONDS = 2400, 7200


def utc_now():
    return datetime.now(timezone.utc).isoformat()

def run_pilot(repo_root, authorization_path, run_root):
    root, out = Path(repo_root).resolve(), Path(run_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    decision = inspect_authorization(root, authorization_path)
    write_json(out / "preflight.json", decision)
    if decision.get("authorized") is not True:
        raise PermissionError("Pilot authorization denied")

    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA must not be initialized for C8")
    initial = flatten_parameters(__import__(
        "lawless.controllers.memory", fromlist=["MemoryController"]
    ).MemoryController(seed=7)).detach().cpu()
    deadline, started = time.monotonic() + WALL_SECONDS, utc_now()
    accounting = PilotAccounting()

    records, summaries, heldout = [], [], []
    terminal, error = "completed", None
    try:
        for name in METHODS:
            optimizer = build_optimizer(name, initial.clone(), POPULATION,
                                       SIGMA, OPT_SEED)
            adapter = PilotAdapter(deadline, MAX_STEPS)
            method_start = time.monotonic()
            candidate_count, episode_count = 0, 0

            for generation in range(GENERATIONS):
                adapter.check()
                candidates = optimizer.ask()
                fitnesses, per_candidate = [], []
                for candidate in candidates:
                    accounting.candidate_start()
                    score, details = adapter.evaluate(
                        candidate, SEARCH_SEEDS, accounting.episode_start)
                    fitnesses.append(score); per_candidate.append(details)

                optimizer.tell(candidates, fitnesses)
                candidate_count += len(candidates)
                episode_count += len(candidates) * len(SEARCH_SEEDS)
                records.append({"method": name, "generation": generation + 1,
                    "candidate_evaluations": candidate_count,
                    "episodes_total": episode_count,
                    "best_fitness": float(max(fitnesses)),
                    "mean_fitness": float(sum(fitnesses)/len(fitnesses)),
                    "best_so_far": float(optimizer.best_fitness)})

            summaries.append({"method": name,
                "candidate_evaluations": candidate_count,
                "episodes": episode_count, "candidate_budget": 60,
                "episode_budget": 180, "runtime_seconds": time.monotonic()-method_start,
                "best_fitness": float(optimizer.best_fitness),
                "best_vector": optimizer.best_vector.tolist(),
                "seed_schedule": list(SEARCH_SEEDS), "seed": OPT_SEED,
                "generations": GENERATIONS, "population": POPULATION,
                "sigma": SIGMA})

            accounting.candidate_start()
            score, details = adapter.evaluate(
                optimizer.best_vector, HELDOUT_SEEDS, accounting.episode_start)
            heldout.append({"method": name, "score": score,
                "seeds": list(HELDOUT_SEEDS), "episodes": details})

    except PilotTimeout as exc:
        terminal, error = "timed_out", str(exc)
    except KeyboardInterrupt:
        terminal, error = "interrupted", "KeyboardInterrupt"
    except Exception as exc:
        terminal, error = "failed", traceback.format_exc()

    search_complete = (len(summaries) == 3 and len(records) == 30
        and all(s["candidate_evaluations"] == 60 for s in summaries))
    c1 = out / "c1"
    c1.mkdir()
    c1_complete = False
    if search_complete:
        from lawless.optimization.compare import write_comparison
        write_comparison({"records": records, "summaries": summaries,
            "manifest": planned_manifest()}, c1)
        c1_complete = True

    supplement = {"schema_version": 1, "manifest_sha256": planned_identity(),
        "terminal_status": terminal, "error": error,
        "candidate_starts": accounting.candidates,
        "episode_starts": accounting.episodes,
        "search_complete": search_complete, "c1_complete": c1_complete,
        "heldout": heldout, "started_utc": started, "ended_utc": utc_now()}
    write_json(out / "pilot_supplement.json", supplement)

    if c1_complete:
        from lawless.optimization.validate_compare import validate_comparison
        c2 = validate_comparison(c1, out / "c2" / "validation_report.json")
        c2["valid"] = c2.get("passed") is True
        write_json(out / "c2" / "validation_report.json", c2)

        from lawless.optimization.provenance import create_provenance, verify_provenance
        p3 = out / "c3" / "provenance_manifest.json"
        p3.parent.mkdir(parents=True, exist_ok=True)
        create_provenance(root, c1, p3)
        p3_report = verify_provenance(p3, root, c1)
        write_json(out / "c3" / "verification_report.json", p3_report)

        from lawless.optimization.execution_receipt import create_execution_receipt
        from lawless.optimization.execution_receipt import verify_execution_receipt
        c5 = out / "c5"; c5.mkdir(exist_ok=True)
        receipt_path = c5 / "execution_receipt.json"
        paths = ["manifest.json", "summary.json", "metrics.json",
                 "summary.csv", "metrics.csv"]
        create_execution_receipt(repo_root=root, evidence_dir=c1,
            authorization_path=authorization_path, artifact_paths=paths,
            consumed_budget=accounting.candidates, started_utc=started,
            ended_utc=utc_now(), terminal_status=terminal,
            receipt_path=receipt_path)

        report = verify_execution_receipt(receipt_path=receipt_path,
            repo_root=root, evidence_dir=c1,
            authorization_path=authorization_path,
            c2_report_path=out / "c2" / "validation_report.json",
            provenance_path=p3)
        write_json(c5 / "receipt_verification_report.json", report)

    else:
        write_json(out / "incomplete_run.json", {
            "terminal_status": terminal, "error": error,
            "search_complete": False,
            "candidate_starts": accounting.candidates,
            "episode_starts": accounting.episodes,
            "manifest_sha256": planned_identity()})

    return {"terminal_status": terminal, "search_complete": search_complete,
        "candidate_starts": accounting.candidates,
        "episode_starts": accounting.episodes,
        "heldout": heldout, "error": error,
        "run_root": str(out)}
