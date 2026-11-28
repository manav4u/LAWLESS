import csv
import json
import time
from pathlib import Path
import torch
from .ga import GA
from .es import OpenAIES
from .cmaes import CMAES


METHODS = ("GA", "OpenAI-ES", "CMA-ES")




def build_optimizer(name, initial, population, sigma, seed):
    if name == "GA":
        return GA(initial, population=population, sigma=sigma, seed=seed)

    if name == "OpenAI-ES":
        return OpenAIES(initial, population=population, sigma=sigma,
                        learning_rate=0.01, seed=seed)
    if name == "CMA-ES":
        return CMAES(initial, population=population, sigma=sigma, seed=seed)
    raise ValueError(f"unknown optimizer: {name}")




def compare_optimizers(initial, evaluator_factory, seeds, generations=10,
                      population=8, sigma=0.1, seed=7):
    seeds = tuple(int(s) for s in seeds)
    if not seeds or generations < 1 or population < 2:
        raise ValueError("seeds, generations, and population invalid")

    initial_state = initial.clone()
    records, summaries = [], []
    candidate_budget = generations * population
    episode_budget = candidate_budget * len(seeds)


    for name in METHODS:
        optimizer = build_optimizer(name, initial.clone(), population,
                                   sigma, seed)
        evaluator = evaluator_factory(seeds)
        started = time.perf_counter()
        candidate_evals, episodes = 0, 0

        for generation in range(generations):
            candidates = optimizer.ask()
            fitnesses = optimizer.evaluate(evaluator, candidates)
            if len(fitnesses) != population:
                raise ValueError("evaluator returned unexpected fitness count")
            optimizer.tell(candidates, fitnesses)
            candidate_evals += len(candidates)
            episodes += len(candidates) * len(seeds)
            records.append({
                "method": name, "generation": generation + 1,
                "candidate_evaluations": candidate_evals,
                "episodes_total": episodes,

                "best_fitness": float(max(fitnesses)),
                "mean_fitness": float(sum(map(float, fitnesses))
                                      / len(fitnesses)),
                "best_so_far": float(optimizer.best_fitness),
            })
        elapsed = time.perf_counter() - started
        summaries.append({
            "method": name, "candidate_evaluations": candidate_evals,
            "episodes": episodes, "candidate_budget": candidate_budget,
            "episode_budget": episode_budget, "runtime_seconds": elapsed,

            "best_fitness": float(optimizer.best_fitness),
            "best_vector": optimizer.best_vector.tolist(),
            "seed_schedule": list(seeds), "seed": seed,
            "generations": generations, "population": population,
            "sigma": sigma,
        })
    assert torch.equal(initial, initial_state)
    return {"records": records, "summaries": summaries,
            "manifest": {"methods": list(METHODS), "seeds": list(seeds),
                         "generations": generations, "population": population,
                         "sigma": sigma, "seed": seed,
                         "candidate_budget": candidate_budget,
                         "episode_budget": episode_budget}}

def write_comparison(result, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(result["manifest"], indent=2), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(result["summaries"], indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(
        json.dumps(result["records"], indent=2), encoding="utf-8")

    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        rows = result["summaries"]
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        rows = result["records"]
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def run_comparison(initial, evaluator_factory, seeds, output_dir,
                   generations=10, population=8, sigma=0.1, seed=7):
    result = compare_optimizers(
        initial, evaluator_factory, seeds, generations,
        population, sigma, seed)
    write_comparison(result, output_dir)
    return result
