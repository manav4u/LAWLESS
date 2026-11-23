# P1.8 CUDA Memory and Tensor-Shape Evidence - B16/N256

Status: Passed - bounded CUDA memory/shape probe
Evidence date: 2026-11-23
Pinned source commit: 0a6b409e9433a428cb6b34e02f0cd96c7cb85b6a
Hardware: NVIDIA Tesla T4
Runtime: Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0

## Scope

This evidence records one bounded representative-scale CUDA execution using the existing public benchmark configuration factory and public environment APIs.

Batch: B=16 worlds

Candidates per world: N=256

Candidate slots: 4096

Seed: 7

Requested dtype: torch.float32

Device: cuda:0

Execution: actual public reset, followed by exactly two actual public step calls

Actions: zeros, shape [16, 256, 2]

The probe intentionally uses the existing Colab clone at /content/LAWLESS_cuda_smoke_0a6b409e9433 and is an evidence script, not a general-purpose benchmark runner. It was executed outside the repository. It does not modify production source.

## Result

Passed. The child CUDA probe returned exit code 0, reported status passed, and confirmed exactly two completed public step calls.

The parent verified the pinned commit and clean checkout before and after execution. Production source hashes and the benchmark factory source hash matched across preflight and postflight. The imported reset and step modules resolved inside the pinned clone.

The exact probe script is archived as:

benchmarks/cuda_memory_shape_probe.py

Bytes: 33,723

SHA-256: 8bb43c3d3d3eaaa4b9fbad82a2004caf8601bce7e523fbea0d73533983e8ea2a

The exact raw evidence archive is T4_memory_shape.zip.

Bytes: 16,754

SHA-256: c48990306a7f31176aa64795112a47b48ee672804dd37a5610bb074a638fa322

The archive hash and byte count were calculated from the completed ZIP and printed outside its contents. No self-referential archive hash or stale embedded ZIP metadata is used.

## Tensor and geometry checks

The probe recursively audited reachable tensors after reset and after the two steps, recording tensor paths, shapes, dtypes, devices, finiteness, and storage sizes.

Reset audit: 344 tensor paths; 344 unique tensor objects.

Final audit: 364 tensor paths; 364 unique tensor objects.

All audited floating-point and complex tensors were finite.

Reset observations: [16, 256, 23], float32, CUDA.

Actions: [16, 256, 2], float32, CUDA.

Per-world physics position: [256, 2], float32, CUDA.

Per-world spawn indices: [256], int64, CUDA.

Per-world generated spawn positions: [1, 2], float32, CUDA.

Rewards: [16, 256], float32, CUDA.

Terminated and truncated: each [16, 256], bool, CUDA.

All 16 worlds reached step_count == 2.

The geometry checks found 16 distinct generated-world objects, one per world. Each world's 256 candidate agents used spawn index zero, referring to that world's single generated spawn. This is evidence of the checked layout and spawn-index contract, not a claim that candidate agents each received independent generated layouts.

## CUDA allocator measurements

All values below are allocator measurements in bytes, not total device VRAM usage.

| Measurement | Allocated | Reserved |
| --- | ---: | ---: |
| Baseline before reset | 0 | 0 |
| Immediately after reset | 756,736 | 6,291,456 |
| Reset-phase peak | 5,146,624 | 6,291,456 |
| Immediately after two steps | 983,552 | 8,388,608 |
| Two-step phase peak | 6,275,072 | 8,388,608 |

Reset peak statistics were captured immediately after public reset and before the recursive reset audit.

The step-phase peak window began after reset-audit cleanup. The actions tensor was allocated inside that window and is included in its peak. The raw reserved-memory peak may include cached allocator blocks retained from reset and its audit. It must not be interpreted as isolated incremental step-only reserved-memory demand.

These values describe PyTorch CUDA allocator state and peaks for this bounded execution. They are not total GPU memory consumption, a full-episode memory profile, or a performance result.

## Integrity and reproducibility

The parent receipt confirms:

Preflight and postflight HEAD: 0a6b409e9433a428cb6b34e02f0cd96c7cb85b6a.

Checkout clean before and after.

Production source unchanged.

Exact benchmark factory source unchanged.

Child status passed.

Child return code 0.

Public step calls completed: 2.

Postflight passed with no postflight errors.

The benchmark factory was sourced from benchmarks/environment_benchmark.py and matched the supplied create_config(seed) implementation. Its SHA-256 was eabbbf393395cbc0b65fa45b146abd710a72f7038e6210a2eeb69fd880c1bc88.

The probe script, child receipt, parent receipt, and raw child output are retained in T4_memory_shape.zip.

### Full local console suite

The full local console suite completed successfully on the local checkout:

- Result: **1,173 passed, 137 skipped, 34 warnings**
- Skips: all 137 were genuine CUDA-unavailable skips
- Duration: 114.21 seconds
- Exit code: 0
- No new test code was introduced in this increment.
- Raw console output: `local_console_suite.txt`
- Captured process exit status: `local_console_exit.json`

This is local CPU-suite evidence. It does not add CUDA coverage or close any remaining original P1.8 GPU correctness, transfer-audit, or performance gates.

## Limitations

This evidence covers reset and two public transitions only. It does not establish:

Full 40-second episode behavior or performance.

Full CUDA component timings or episode throughput.

CPU-versus-T4 equivalence at B16/N256.

Absence of unintended CPU transfers across the relevant execution path.

Total device VRAM usage.

Overall P1.8 acceptance or closure.

No production optimization or production-source change was made for this evidence.

## Recommended next P1.8 increment

Continue with the original P1.8 performance-evidence scope: measure the actual CUDA component timings and a complete 40-second CUDA episode using the existing benchmark configuration and public execution path. Keep the measurement bounded and separate from production changes. Record initialization, reset, physics/collision, lidar, observations, complete-step timing, and full-episode timing where the existing benchmark permits.

The outstanding no-unintended-CPU-transfer and broader CPU/T4 correctness evidence remain separate open items. Do not treat this memory/shape pass as satisfying them or as a new acceptance protocol.
