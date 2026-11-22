# LAWLESS P1.8-A2.1, Controlled CPU Episode Correctness Evidence

## Status

**A2.1 CPU correctness/stress substage: PASS**

This checkpoint records a successful controlled full-episode CPU stress run and the final candidate's test results. It is bounded correctness and accounting evidence only. It is not a throughput benchmark, a GPU result, or acceptance of the overall P1.8-A protocol.

The tested checkout was based on parent revision `a181451` and was dirty. The tested source identities are recorded by SHA-256 in `source_manifest.json`; this checkpoint does not describe the tested checkout as clean or identify it with a later commit.

## Locked fixture

The controlled run used:

* Seed: `7`
* Batch: `1` world
* Candidates per world: `1`
* Device: CPU
* Episode duration: `40.0 s`
* Timestep: `1/60 s`
* Required episode length: `2,400` public transitions
* Gravity magnitude: `4.0 m/s²`
* Required gravity shift: transition `1,200`, at `20.0 s`
* Initial geometry: 20 obstacles and 8 cores

The fixture used the production environment step path and the A2.1 hold-position controller. The controller target was captured from the actual reset state. No production physics or initial-state changes were made for this checkpoint.

## Full-episode result

The final seed-7 B1/N1 receipt completed with status `completed` and termination reason `episode_deadline` at transition 2,400.

| Measure | Recorded result |
| --- | --- |
| Public step calls | 2,400 |
| World transitions | 2,400 |
| Candidate-slot transitions | 2,400 |
| Alive-agent transitions | 2,400 |
| Newly completed worlds | 1 |
| Gravity shift | Transition 1,200 |
| Gravity shift elapsed time | 20.0 s |
| Geometry inventory checks | 2,400 |
| Per-world geometry checks | 2,400 |
| Last geometry check | Transition 2,400 |
| Geometry validation complete | Yes |

The gravity shift was reported and installed in the returned world state. Its configured direction was `+1`; gravity changed from `[4, 0]` to `[0, 4]`. The persistent shift flag was true and the actual world step counter was 1,200 at the shift.

The initial and final geometry inventories matched: 20 obstacles and 8 cores. Geometry was checked after every successfully returned transition, after that transition had been installed and recorded in accounting.

The run recorded two collision contacts. Boundary escapes, unresolved contacts, failures, and energy deaths were all zero. At the episode deadline, the final agent remained physically alive and was truncated by the production episode deadline.

## Final candidate test results

The final candidate was run twice: first through the combined focused benchmark tests, then through the full unfiltered CPU suite.

| Test gate | Result |
| --- | --- |
| Four combined benchmark test modules | 118 passed |
| Combined benchmark test runtime | 42.30 s |
| Full unfiltered CPU suite | 1,055 passed |
| Genuine CUDA skips | 133 |
| Existing warnings | 34 |
| Full suite runtime | 57.15 s |

The 133 skips are genuine CUDA-dependent skips in the CPU environment. They are not CUDA validation results.

## Source identity

The tested source and test-file SHA-256 digests are:

| File | SHA-256 |
| --- | --- |
| `benchmarks/environment_benchmark.py` | `eabbbf393395cbc0b65fa45b146abd710a72f7038e6210a2eeb69fd880c1bc88` |
| `tests/test_environment_benchmark_a21.py` | `7850dc123b3569c07803483aefbd09143d8aa20c3993271d99cf09088fd5d284` |
| `tests/test_environment_benchmark_a21_regressions.py` | `1dd71a91d2cb1c92e65d21ca8503641dd2292915542762d3512612a2a7394054` |

`source_manifest.json` records these digests, the frozen P0 Python-source identity, and the original and published log hashes. The frozen 20-file Python inventory matches the protected P0 identity. No production source was changed for A2.1.

The tested receipt's source identity records parent revision `a181451` and `dirty: true`. That is intentional: the tested checkout must not be represented as clean.

## Evidence files

This directory contains:

* `full_episode.json`, final successful seed-7 B1/N1 run receipt.
* `focused_tests.txt`, original focused benchmark-test output.
* `cpu_suite.txt`, original full unfiltered CPU-suite output.
* `source_manifest.json`, tested source/test digests, frozen-source identity, and original/published log hashes.
* `negative_receipts/`, retained negative-case receipts.

Root-path prefixes in warning locations in the published logs have been removed for portability. This is a path-only presentation normalization; it does not change test outcomes or warning content. Original and published log hashes are retained in `source_manifest.json` so the transformation is disclosed and auditable.

The A1 evidence and README remain separate. This A2.1 checkpoint does not replace or rewrite the A1 evidence.

## Scope and limitations

This result demonstrates one controlled CPU episode at B1/N1 with seed 7. It does not establish:

* Performance or throughput, including any extrapolation to larger batches or candidate counts.
* A representative distribution across seeds or workloads.
* Learned-controller or research-controller performance or validity.
* GPU correctness, GPU memory use, or free Tesla T4 performance.
* Component-level performance attribution.
* Compatible resumability.
* Vectorization equivalence or a decision to vectorize.
* Full P1.8-A protocol acceptance.

No hosted-CI result is claimed here. No main-branch merge or overall P1.8-A acceptance is implied.

## Next work

Proceed to A2.2 as a separate, source-grounded validation substage. Preserve this A2.1 checkpoint unchanged. The next stage must distinguish uninstrumented timing from accounting and profiling overhead, and must not convert this correctness run into a throughput claim.
