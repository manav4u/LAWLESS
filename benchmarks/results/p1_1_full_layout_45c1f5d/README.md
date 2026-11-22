# P1.1 Full-Layout World Generation - Validation Evidence

## Status

**P1.1 world-invariant gate: met for candidate `45c1f5d887f969463df16b38517b77481c7b4400`.**

This decision is based on the reported CPU suite, free-Colab T4 suite, P0 regression validation, candidate CI result, and benchmark instrumentation checks below. It applies to the P1.1 world-generation and geometry invariants only. It does **not** mean that all of P1 is complete, that the candidate has been promoted to `main`, or that P1.8 performance acceptance has been met.

- Evidence target: `45c1f5d887f969463df16b38517b77481c7b4400`
- Historical `main` status at evidence capture: `96ec044`
- P0 remains frozen at tag `p0-validated`, commit `d82d8e1d910285b4d95ca7a64ad71d4a2437a883`
- Candidate CI: [run 45006834337](https://github.com/manav4u/LAWLESS/actions/runs/45006834337)

The historical `main` reference records the branch status when this evidence was captured. It makes no claim about the current branch state after publication of this document.

## Scope validated

The full-layout generator creates a continuous 2D world containing:

- Static circular obstacles.
- A configurable number of spawn positions.
- One extraction region.
- Energy-core positions and radii.

The generator uses one private CPU `torch.Generator`. It preserves the established obstacle-generation helper and its random-draw order, then generates spawn positions, extraction, and cores. Geometry is transferred to the requested device after sampling.

The validated invariants include containment and strict positive clearance between generated geometry categories. Tangency is rejected during generation. Generation uses finite per-category attempt budgets and reports deterministic `WorldGenerationError` details on exhaustion; it does not return a partial world or fabricate placements for unvisited indices.

`GeneratedWorld` remains a geometry/provenance record. This batch does not implement episode core ownership or depletion. When episode bookkeeping is added, core collection must remain independently tracked per agent; this generator does not establish competitive shared depletion.

## Test and regression evidence

| Validation | Result |
|---|---:|
| CPU, `python -m pytest` | 391 passed, 16 CUDA skips |
| Free Colab Tesla T4, `pytest -q` | 407 passed, 0 skipped |
| P0 validator, CPU | 2,400 steps / 40 seconds passed |
| P0 validator, T4 | 2,400 steps / 40 seconds passed |
| Candidate CI | Green |

The Colab run used Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0, and a free Tesla T4 allocation.

The test import loads `scripts/benchmark_full_layout.py` using a path relative to the test file. This supports both the repository's console command, `pytest -q`, and `python -m pytest`, without adding repository-root entries to `sys.path`, installing `scripts` as a package, or changing product packaging.

The historical obstacle-only RNG golden baselines remain unchanged. The full-layout tests also verify the benchmark instrumentation against ordinary generation: returned layouts and final private-generator states match exactly. Instrumentation runs separately from timed runs.

### Early obstacle-exhaustion accounting

The corrected instrumentation reports only obstacle candidates actually attempted. It preserves the original `WorldGenerationError` and does not invent counts for obstacle indices that were never reached.

Two failure cases were checked:

- The normal 20-obstacle, seed-7, float32 case with obstacle budget 1 exhausts at obstacle index 4, reporting 5 total attempts.
- The deterministic 2×2 m case with 20 requested obstacles, obstacle radius 1 m, and budget 3 exhausts at index 1. It reports 4 total attempts: obstacle 0 used 1 attempt and obstacle 1 used 3.

All seven benchmark-counter tests pass as part of the reported candidate test evidence.

## Benchmark configuration

Raw JSONL files are stored alongside this README:

- `cpu_torch_cpu.jsonl`
- `colab_cpu_torch_cu130.jsonl`
- `colab_cuda_t4.jsonl`
- `early_obstacle_exhaustion_cpu.jsonl`

Normal benchmark cases use:

- 20 obstacles, 4 spawns, and 8 cores.
- Seeds 7 and 42.
- float32 and float64.
- Three ordinary, uninstrumented timing repeats per case.
- Per-category attempt budgets of 10,000.
- CUDA synchronization outside/around the ordinary timed calls.
- Separate instrumentation runs, excluded from ordinary timing.

The attempt totals below are the observed totals for the recorded layouts. They are not fixed attempt-count requirements for arbitrary configurations.

## Recorded median generation times

All values are milliseconds per single generated layout. Each is the median of three ordinary uninstrumented repeats.

| Runtime | Seed | Dtype | Median (ms) | Obstacle attempts | Spawn attempts | Extraction attempts | Core attempts |
|---|---:|---|---:|---:|---:|---:|---:|
| CPU, Torch 2.14.1+cpu | 7 | float32 | 2.5193540004693205 | 22 | 4 | 1 | 8 |
| CPU, Torch 2.14.1+cpu | 7 | float64 | 2.089690000502742 | 20 | 4 | 1 | 8 |
| CPU, Torch 2.14.1+cpu | 42 | float32 | 2.411667999695055 | 22 | 4 | 1 | 8 |
| CPU, Torch 2.14.1+cpu | 42 | float64 | 2.165416999559966 | 21 | 4 | 1 | 9 |
| Colab CPU, Torch 2.14.1+cu130 | 7 | float32 | 6.528045000095517 | 22 | 4 | 1 | 8 |
| Colab CPU, Torch 2.14.1+cu130 | 7 | float64 | 5.318482000006952 | 20 | 4 | 1 | 8 |
| Colab CPU, Torch 2.14.1+cu130 | 42 | float32 | 6.090473000085694 | 22 | 4 | 1 | 8 |
| Colab CPU, Torch 2.14.1+cu130 | 42 | float64 | 5.4012619999639355 | 21 | 4 | 1 | 9 |
| Colab Tesla T4, Torch 2.14.1+cu130 | 7 | float32 | 9.327044999963618 | 22 | 4 | 1 | 8 |
| Colab Tesla T4, Torch 2.14.1+cu130 | 7 | float64 | 8.018943000024592 | 20 | 4 | 1 | 8 |
| Colab Tesla T4, Torch 2.14.1+cu130 | 42 | float32 | 8.59930000001441 | 22 | 4 | 1 | 8 |
| Colab Tesla T4, Torch 2.14.1+cu130 | 42 | float64 | 8.164145000023382 | 21 | 4 | 1 | 9 |

The first CUDA sample is approximately 155 ms and remains in the raw data. It has not been hidden or substituted. The table reports medians, not a claim that every call takes the median time.

## Timing limitations

These measurements are small single-layout samples with only three ordinary repeats per case. They are useful as reproducible baseline observations for this generator and runtime configuration, not as a stable performance characterization.

In particular:

- They are not P1.8 throughput evidence.
- They do not measure batched episode stepping, agent-steps per second, episodes per second, or full 40-second episodes.
- They do not establish a GPU speedup. In these samples, the T4 medians are higher than the corresponding CPU medians; no broader performance conclusion should be inferred from four cases and three repeats.
- CUDA synchronization is performed outside/around the ordinary timed calls, while instrumentation is run separately. Instrumented timings must not be compared with ordinary timings.
- The first approximately 155 ms CUDA sample is retained in the raw JSONL evidence.
- Warmup and outlier effects remain limitations of these short runs. Three repeats are insufficient to characterize their distribution or establish stable latency.

P1.8 must use representative workloads, adequate warmup, repeated measurements, and the actual environment-step pipeline before any throughput or acceleration claim is made.

## Reproduction

Run commands from the repository root. Install the project and development dependencies first:

```bash
python -m pip install -e '.[dev]'
```

Run the complete test suite:

```bash
pytest -q
```

or:

```bash
python -m pytest
```

Run the P0 regression validator:

```bash
python benchmarks/validate_p0.py
```

The P0 validator executes 2,400 steps over 40 simulated seconds and reports numerical deviations on CUDA. The reported CPU and free-T4 validations both passed.

### Reproduce ordinary benchmark cases

The benchmark CLI supports `--device {cpu,cuda}`, `--seeds`, `--dtypes`, `--repeats`, `--obstacles`, `--spawns`, `--cores`, and `--max-attempts`.

Local CPU:

```bash
python scripts/benchmark_full_layout.py \
 --device cpu \
 --seeds 7 42 \
 --dtypes float32 float64 \
 --repeats 3 \
 --obstacles 20 \
 --spawns 4 \
 --cores 8 \
 --max-attempts 10000
```

Colab CPU:

```bash
python scripts/benchmark_full_layout.py \
 --device cpu \
 --seeds 7 42 \
 --dtypes float32 float64 \
 --repeats 3 \
 --obstacles 20 \
 --spawns 4 \
 --cores 8 \
 --max-attempts 10000
```

Colab Tesla T4:

```bash
python scripts/benchmark_full_layout.py \
 --device cuda \
 --seeds 7 42 \
 --dtypes float32 float64 \
 --repeats 3 \
 --obstacles 20 \
 --spawns 4 \
 --cores 8 \
 --max-attempts 10000
```

These commands reproduce the ordinary benchmark configuration. Preserve the JSONL output from each runtime alongside this README using the corresponding filenames:

- Local CPU: `cpu_torch_cpu.jsonl`
- Colab CPU: `colab_cpu_torch_cu130.jsonl`
- Colab T4: `colab_cuda_t4.jsonl`
- Early obstacle exhaustion: `early_obstacle_exhaustion_cpu.jsonl`

The early-exhaustion JSONL is a separate failure-accounting record, not an ordinary timing run. Keep it alongside the three ordinary runtime records.

The reported evidence targets the exact candidate commit `45c1f5d887f969463df16b38517b77481c7b4400`. To reproduce the reported candidate results, check out that exact commit before running the commands in matching environments. The commands above run the benchmark and print its results; preserve the resulting raw output as the JSONL evidence files.

Do not replace outliers or selectively remove samples to make medians appear more favorable.

## Gate decision and next stage

The P1.1 acceptance target is reproducible continuous-world generation with valid geometry. The candidate has passed the reported CPU and T4 test suites, P0 regression validation on both devices, candidate CI, seeded-layout/RNG checks, and explicit early-failure accounting.

**Decision: P1.1 world-invariant gate is met on the validation candidate.**

This gate does not certify collision response, agent lifecycle, objectives during an episode, lidar, observations, episode execution, or vectorized environment performance. Those remain later stages. This document does not assert that `main` was promoted.

The free Colab T4 session used for the reported validation is now disconnected and deleted; Manage sessions shows no active sessions. The reported T4 evidence remains historical evidence from that completed run, not a claim of a currently active GPU session.

The next planned stage is P1.2 agent embodiment and episode bookkeeping. Implementation must wait for review of the exact current config, physics-state, and `GeneratedWorld` APIs. P0 physics remains authoritative and unchanged. Per-world damping remains a later, explicit P1.7 integration.
