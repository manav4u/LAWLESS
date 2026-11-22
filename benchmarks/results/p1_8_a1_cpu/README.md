# P1.8-A1 CPU checkpoint (incomplete)

## Status

**A1 CPU functional and accounting checkpoint: recorded.**

**Overall P1.8-A: INCOMPLETE.**

This directory records a bounded CPU validation of the environment benchmark candidate. It is an explicitly incomplete feature-branch checkpoint. It is not a P1.8-A acceptance report and must not be interpreted as one.

## Source and preservation

The candidate was tested on the feature branch `validation/environment-benchmarks`.

At validation time, the reported live remote refs `main` and `validation/episodes` both pointed to:

`ac7e0471ae314460d6f72036a33f9a2e3a198bc5`

The candidate files were local only; no candidate changes were pushed.

The frozen P0 tag's remote commit was:

`d82d8e1d910285b4d95ca7a64ad71d4a2437a883`

All 20 Python files present at frozen P0 were verified byte-for-byte against the tag. P0 and the accepted P1.7 source were not modified as part of this checkpoint.

Candidate file SHA-256 digests:

| File | SHA-256 |
|---|---|
| `benchmarks/environment_benchmark.py` | `ebae7ba6e709e152642b2b7e34015c62a11042b5e6c068ddf6666c2ba8f15e9c` |
| `tests/test_environment_benchmark.py` | `15960a0fec0fb074ed70c239d0bc3f77125f9ca9678c452576722bffc17f0641` |
| `tests/test_environment_benchmark_regressions.py` | `b7279e0d3ce9e299181d3e736c97e30741992ea25a3971abe5f924b3e69925a4` |

## Test evidence

The existing 20 focused tests were preserved.

- Combined focused tests: **97 passed**, zero failures, 8.10 seconds.
- Full unfiltered CPU suite: **1034 passed**, **133 CUDA skips**, zero failures, 34 existing warnings, 24.84 seconds.
- The CUDA skips are recorded as skips; this run provides no GPU execution or GPU performance evidence.

Detailed captured output is stored in `focused_tests.txt` and `cpu_suite.txt`.

## Bounded CPU functional run

The recorded invocation used:

```text
--mode pilot
--device cpu
--batch 16
--candidates 256
--repetitions 1
--max-steps 2
--post-reset-wall-budget 30
--seed 7
```

The run executed:

- 2 step calls
- 32 world-step transitions across 16 worlds
- 8,192 candidate-slot transitions
- 8,192 alive-agent transitions
- 20 obstacles and 8 cores in each of the 16 worlds

Its actual status was **`bounded_incomplete`**.

This establishes bounded functional execution, workload accounting, and the reported geometry shape for this invocation. It is not a full episode, a controlled 2,400-transition stress fixture, a throughput baseline, or evidence of performance improvement. The 30-second post-reset budget is an invocation limit, not a claim that the run consumed that duration.

The exact invocation and run record are retained in `invocation.json` and `run_000.json`.

## Evidence inventory

This directory may contain:

- `README.md` : scope, status, and interpretation of this checkpoint
- `cpu_suite.txt` : full unfiltered CPU suite output
- `focused_tests.txt` : focused test output
- `source_manifest.json` : source identities and file digests
- `invocation.json` : actual B16/N256 invocation metadata
- `run_000.json` : actual bounded run record

All filenames are relative to this directory. Do not add absolute executable, workspace, home-directory, or other private machine paths to these records.

## Explicitly not established

This checkpoint does not establish:

- completion of a full 2,400-transition episode or controlled full-episode stress fixture
- sustained or uninstrumented step throughput
- component-level profiling results
- GPU execution, peak GPU memory, or Free Tesla T4 performance
- compatible pause/resume behavior
- CPU/GPU equivalence or vectorization equivalence
- P1.8-A protocol acceptance

## Next protocol work

The remaining protocol must be completed as separately identified evidence. In particular, it requires a controlled full-episode stress fixture with nonempty geometry and real gravity/gravity-shift behavior; uninstrumented timing separated from accounting; component profiling; GPU memory measurement; compatible resumability; and controlled CPU and Free Tesla T4 baselines. Equivalence gates must pass before any vectorization decision.

Until those gates are independently run, recorded, and reviewed, overall P1.8-A remains **INCOMPLETE**.
