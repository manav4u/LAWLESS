# P1.7 Continuous 2D Episodes - Source Validation Receipt

## Status and provenance

P1.7 source validation passed on the exact candidate source commit:

- Source commit: `fef4a95d6ced1464c0ceae283cfea5b9284398fa`
- Source branch at validation: `validation/episodes`
- Source commit URL: https://github.com/manav4u/LAWLESS/commit/fef4a95d6ced1464c0ceae283cfea5b9284398fa
- Source CI: passed
- Source CI run: https://github.com/manav4u/LAWLESS/actions/runs/45042415737
- Source CI job: https://github.com/manav4u/LAWLESS/actions/runs/45042415737/job/137269576801
- Main at the time of source validation: `c0ecad36767f8c79644fe6753f32fdb232a5fe13`
- Frozen P0 tag: `p0-validated`
- Frozen P0 tag commit: `d82d8e1d910285b4d95ca7a64ad71d4a2437a883`

This receipt documents validation of the P1.7 source commit. It does not assert successful CI for a documentation-only successor.

## Measured validation results

### CPU

- Full unfiltered pytest suite: **937 passed, 133 genuine CUDA skips, zero failures**
- Warnings: 34 prior collision-fixture warnings
- Runtime: 16.43 seconds
- Python: 3.10.12
- PyTorch: 2.14.1+cpu

### Actual free Tesla T4

- Full unfiltered pytest suite: **1,070 passed, zero skips, zero failures**
- Warnings: 34 prior collision-fixture warnings
- Runtime: 138.50 seconds
- Python: 3.13.16
- PyTorch: 2.14.1+cu130
- CUDA: 13.0
- GPU: Tesla T4
- GPU memory: 14.56 GiB

The zero-skip T4 suite result does not mean all 1,070 tests instantiated CUDA tensors. It means the full suite ran on the T4-equipped runtime, including the CUDA-specific tests that require CUDA availability.

## P0 validator on the actual T4

The frozen P0 validator completed successfully on the actual T4 at the validated candidate HEAD.

- Validator: 2,400 transitions
- Duration: 40.0 seconds at the default timestep
- Return code: 0

Observed maximum absolute CPU/CUDA deviations over the long rollout:

| Field | Maximum absolute deviation |
| --- | --- |
| Position | 0.0001220703125 |
| Velocity | 0.000003814697265625 |
| Angle | 0.0 |
| Angular velocity | 0.0 |
| Energy | 0.0 |

These are observational long-rollout values, not new numerical tolerance gates. They do not change or relax frozen P0 acceptance tolerances.

## Functional API

The P1.7 environment uses functional state-in/state-out operations rather than a stateful `env.step` abstraction.

- Functional reset returns `environment`, `observations`, and `reset_info`.
- Functional step returns the next `environment` plus `observations`, `rewards`, `terminated`, `truncated`, and `info`.

The environment represents B independent worlds, each with N independent candidate agent episode replicas. Within each world, the default generated spawn slot is shared by candidate replicas. Candidates therefore undergo paired evaluation against the same generated world and spawn.

Candidate physics, energy, objectives, and lifecycle masks are independent. Each world owns a shared episode clock, rather than each candidate replica maintaining an independent clock. Candidates do not compete with one another or deplete a shared core resource.

There is no agent-agent competition, collision, or shared core depletion. Each candidate collects and consumes core state independently. The world layout and challenge seed policy are invariant under N: changing candidate batch size does not change the underlying generated world and challenge configuration for a given world seed/index.

## Reward and episode semantics

- Reward is the raw newly-extracted indicator for the transition, with values 0 or 1. It is not shaped reward or final fitness.
- Core collection, extraction, energy, collision/failure, termination and timeout diagnostics are exposed separately through info.
- World-local completion and freezing are independent: completed worlds cannot advance while other worlds continue.
- The default episode is 2,400 transitions at `dt=1/60`, corresponding to 40 seconds.
- Extraction is evaluated before the deadline timeout on the deadline transition.
- The default gravity shift occurs at the boundary between transitions 1200 and 1201: transition 1200 uses the original gravity; the rotation is applied at the end of transition 1200, and transition 1201 uses the rotated gravity.
- The shift direction sign is explicitly randomized between the two possible rotation signs under the seed policy.

## Test coverage represented in source acceptance

The accepted source tests cover:

- Functional API contracts and reset/step output structure.
- B independent worlds and N independent candidate replicas, with shared per-world spawn policy.
- Reproducibility and seed-policy invariance under N.
- Default 2,400-transition episode behavior and deadline semantics.
- Gravity-shift transition boundary and diagnostics.
- Both shift signs on CPU.
- Collision-resolved velocity used for finite-difference acceleration measurements.
- Core collection, actual capped restoration, and independent candidate core lifecycle.
- World-local completion, freezing and all-finished behavior.
- Complete action tensor validation before mutation.
- Observation structure, normalization, privacy and consistency.
- RNG isolation and seed-policy behavior.
- Actual CUDA execution cases and CPU/CUDA parity test coverage.

The CUDA default-duration fixture was corrected before hardware execution, and a CPU timing preflight was added to ensure the fixture's timing configuration was valid. No environment implementation fixes were required after the initial delivery.

The new CPU/CUDA parity test uses `rtol=1e-4, atol=2e-5`. These parity-test tolerances are separate from and do not change the frozen P0 numerical acceptance tolerances.

The actual T4 full-suite result includes the CUDA tests without skips. It does not imply every full-suite test constructs or advances CUDA tensors.

## Validation-runner bookkeeping and closeout

The actual free-T4 validation runner completed the full pytest suite and the frozen P0 validator successfully. Its final whole-working-tree-clean assertion failed afterward because the editable installation generated the ordinary untracked `lawless_sim.egg-info/` package metadata.

This was a runner bookkeeping issue, not a test failure or implementation defect. The original assertion and full runner output are retained in `t4_validation_initial.txt`.

The separate read-only closeout on the same actual free T4 confirmed:

- Exact candidate HEAD remained `fef4a95d6ced1464c0ceae283cfea5b9284398fa`.
- No tracked source changes.
- No staged changes.
- The entire tracked `lawless/physics` file inventory and every frozen physics blob matched `p0-validated` byte-for-byte.
- `benchmarks/validate_p0.py` matched the frozen P0 tag byte-for-byte.
- `p0-validated` still resolves to `d82d8e1d910285b4d95ca7a64ad71d4a2437a883`.
- Five ordinary setuptools metadata files remained untracked under `lawless_sim.egg-info/`. Their package metadata identity was verified.

The closeout did not rerun tests, install anything, delete files, or modify source. The generated metadata was retained until the free runtime was explicitly terminated. The runtime termination was confirmed, with no active sessions remaining.

## Raw evidence

The validation evidence directory contains the following prepared files:

- `cpu_pytest.txt`
- `cpu_runtime.json`
- `t4_validation_initial.txt`
- `t4_closeout.txt`
- `t4_closeout.json`
- `source_ci.json`

The runner's original bookkeeping assertion remains in the raw log. The closeout receipt is separate and does not suppress, rewrite, or reclassify the original runner result.

## Scope boundary

P1.7 implements and validates the continuous 2D episode and observation/transition assembly contract for generated worlds and independent candidate replicas. It does not implement controllers, optimization, training, evolutionary experiments, final fitness aggregation, or research conclusions.

P1.8 performance/vectorization work remains open. The measured test-suite runtimes above are validation runtimes, not environment throughput measurements or a performance benchmark.

This source validation is not overall LAWLESS completion or a research release.