# P1.4 Energy Cores and Extraction - Validation Evidence

## Status

P1.4 energy-core and extraction candidate source validation passed. Final review found no P1.4 source blocker in the supplied implementation and test files.

- Exact tested source commit: `f03704d58500de041bbfad7403bb315878110052`
- Candidate branch: `validation/objectives`
- Candidate source CI: [GitHub Actions run 45023285801](https://github.com/manav4u/LAWLESS/actions/runs/45023285801) - successful
- P1.3 baseline before P1.4 integration: `e59445b64a3295409e6295beaaebd44f3be8d964`
- Frozen P0 tag: `p0-validated` at `d82d8e1d910285b4d95ca7a64ad71d4a2437a883`

The exact source revision tested is `f03704d58500de041bbfad7403bb315878110052`. The documentation and raw validation logs in this directory record the results for that source revision. A documentation-only successor commit does not change the tested source identity or constitute a new source-validation target.

The planned integration is documentation and raw validation logs only relative to the tested P1.4 source. The subsequent fast-forward of `main` is intended to include that exact validated source and its evidence without additional product, test, or validator changes.

Candidate source CI and successor/main CI are separate checks. The successful candidate source CI run above does not establish that CI for the documentation-only successor or integrated `main` has passed. Record the successor/main CI result only after it has actually been checked.

## Runtime

The full CUDA validation used a free Tesla T4 runtime.

| Component | Runtime |
|---|---|
| Python | 3.13.16 |
| PyTorch | 2.14.1+cu130 |
| CUDA | 13.0 |
| GPU | Tesla T4 |
| GPU memory | 14.56 GiB |

No paid compute was used.

## Full test suite

### CPU

| Result | Count |
|---|---:|
| Passed | 717 |
| Genuine CUDA skips | 56 |
| Failed | 0 |

The CUDA-dependent tests were skipped because the CPU run had no CUDA device. These are expected hardware-dependent skips, not test failures.

### Free Tesla T4

| Result | Count |
|---|---:|
| Passed | 773 |
| Skipped | 0 |
| Failed | 0 |
| Runtime | 8.43 seconds |

All tests ran on the T4-equipped runtime.

The T4 test run emitted 34 warnings from existing fixture construction in `tests/test_p13_collision.py`:

- 17 tensor-copy warnings for obstacle centers.
- 17 tensor-copy warnings for obstacle radii.

These warnings are not failures. They are recorded here rather than silently omitted or attributed to new P1.4 product code.

## P0 40-second validator

The P0 validator passed on both CPU and the free Tesla T4: 2,400 steps over 40.0 seconds at the default 1/60-second timestep.

| Field | Maximum absolute CPU/T4 deviation |
|---|---:|
| Position | 0.0001220703125 m |
| Velocity | 0.000003814697265625 m/s |
| Angle | 0.0 |
| Angular velocity | 0.0 |
| Energy | 0.0 |

Both validator runs returned success. These long-rollout deviations are observational evidence, not a new cross-device tolerance gate. The P0 acceptance contract remains the previously established one-step numerical agreement test and exact alive-mask agreement.

The frozen `p0-validated` tag remains at `d82d8e1d910285b4d95ca7a64ad71d4a2437a883`. This P1.4 validation does not move or replace that tag.

## P1.4 coverage reviewed

The supplied candidate source and complete P1.4 test files cover:

- Core-contact geometry using generated-world core centers and radii.
- Per-agent collection, repeated-collection idempotence, and zero-core layouts.
- Energy restoration using current episode energy, including the maximum-energy cap and no-event preservation.
- Eligibility rules for dead, failed, extracted, boundary-escaped, and unresolved-contact agents.
- Use of post-resolution positions and actual P1.3 obstacle-resolution behavior.
- Extraction-radius and episode-deadline inclusive boundaries, including immediately-outside cases.
- Extraction eligibility independent of core collection and remaining energy.
- Revalidation of current geometry, collision disqualification, and deadline during extraction application.
- Preservation of energy, position, objective state, transition measurements, spawn indices, and input ownership.
- Sequential collection followed by extraction while reusing a collision snapshot with stale energy.
- Float32 and float64 coverage, the smallest supported composed-episode batch shape, and CUDA device/CPU parity.

The composed `AgentEpisodeState` contract continues to require at least one positive batch dimension. This work does not weaken that contract or infer scalar composed-episode support from the lower-level `AgentState` API.

## Scope and remaining work

This evidence establishes P1.4 candidate validation only. It does not establish completion of LAWLESS or completion of later milestones.

The following remain **not complete**:

- P1.5 - 8-ray lidar.
- P1.6 - 23-dimensional observations.
- P1.7 - episode execution, including the planned physics shift and termination/report consumption.
- P1.8 - vectorization, parity, and performance work.
- Later controller, optimizer, training, and evolution research phases.

P1.4 does not add controllers, evolution, or training. P1.3 endpoint collision resolution is not swept collision detection and does not guarantee against tunneling.

## Raw evidence files

This directory contains the following raw validation records:

- `cpu_pytest.txt` - full CPU pytest output.
- `t4_pytest.txt` - full free Tesla T4 pytest output.
- `p0_cpu.txt` - CPU P0 2,400-step / 40-second validator output.
- `p0_t4.txt` - Tesla T4 P0 2,400-step / 40-second validator output.
- `runtime.txt` - runtime and hardware details.

The raw files should be retained alongside this README. Their contents, rather than this summary alone, are the underlying execution record.

## Final review

No P1.4 source blocker was identified in the supplied candidate files and validation record. The exact tested source revision is `f03704d58500de041bbfad7403bb315878110052`.

The planned documentation-only integration records this exact validated source and its raw evidence. It does not imply that successor/main CI has passed; that result must be checked and recorded separately.

This is a P1.4 candidate validation result, not a claim that a later documentation-only commit was itself source-tested, that main-branch CI passed, or that the overall research project is complete.
