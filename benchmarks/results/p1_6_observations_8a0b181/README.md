# P1.6 Observation Assembly Validation

## Status

P1.6 observation assembly source validation is complete at exact source
commit `8a0b181c2d0aa7e93ed455fb783160b8db2f92cf` on
`validation/observations`.

The exact tested source revision passed the full CPU suite
(905 passed, 131 genuine CUDA skips, zero failures) and the full
free Tesla T4 suite (1,036 passed, zero skips, zero failures).
Both runs reported 34 existing collision-fixture warnings.

The P0 validator also passed on the same exact source commit on the
actual free Tesla T4 runtime: 2,400 steps over 40.0 seconds.

Source CI completed successfully:
[Source CI run 45034642956](https://github.com/manav4u/LAWLESS/actions/runs/45034642956).

This receipt records P1.6 source validation. It does not claim
overall LAWLESS completion, controller implementation, optimization,
training, evolution, or a research release.

## Scope and source provenance

- Exact tested source: `8a0b181c2d0aa7e93ed455fb783160b8db2f92cf`
- Source branch: `validation/observations`
- Source commit: [8a0b181](https://github.com/manav4u/LAWLESS/commit/8a0b181c2d0aa7e93ed455fb783160b8db2f92cf)
- Source CI: [run 45034642956](https://github.com/manav4u/LAWLESS/actions/runs/45034642956)
- Source CI result: completed successfully
- Previous main revision at the time of source acceptance: `e785ef4`
- Frozen P0 tag: `p0-validated`
- Frozen P0 commit: `d82d8e1d910285b4d95ca7a64ad71d4a2437a883`

The P1.6 source candidate contains the observation normalization and
assembly implementation, sensor exports, and their tests. Its
acceptance preserves all previously accepted source, tests, and
validator behavior. The CPU and T4 results below were measured against
the exact source commit identified above.

The P0 tag remains frozen at its recorded commit. P1.6 does not change
P0 physics, the P0 validator, or the P0 acceptance contract.

## Observation contract

P1.6 assembles a 23-channel observation for each agent from the
composed episode state, the extraction objective, the completed
transition measurements, and the eight-ray lidar result.

The output shape is:

`[*episode_state.batch_shape, 23]`

The observation supports positive, nonempty batch dimensions,
including non-square and multidimensional batches. Output retains the
input floating-point dtype and device.

### Target semantics

The observation target is always `GeneratedWorld.extraction_center`.

The extraction center remains the navigation target throughout the
episode. The implementation does not switch to the nearest
uncollected core. Cores remain separate collection and reward
objectives; they do not replace the extraction-center navigation
target.

Target displacement is expressed in the agent body frame:

- Body +X is forward.
- Body +Y is left.
- Heading is counterclockwise.
- World vectors are rotated by `R(-theta)` into the body frame.

At zero target distance, the target-bearing sentinel is
`sin = 0`, `cos = 1`.

### Exact 23-channel layout

| Channels | Meaning | Normalization or representation |
| --- | --- | --- |
| 0–1 | Target-relative position in body coordinates | Each coordinate divided by 50 m and clipped to `[-1, 1]` |
| 2 | Target distance | Divided by 100 m and clipped to `[0, 1]` |
| 3–4 | Target-bearing sine and cosine | Unit direction representation; zero-distance sentinel `(0, 1)` |
| 5–6 | Linear velocity in body coordinates | Each component divided by 12 m/s and clipped to `[-1, 1]` |
| 7 | Angular velocity | Divided by 4 rad/s and clipped to `[-1, 1]` |
| 8–9 | Heading sine and cosine | Unit direction representation |
| 10–17 | Eight lidar distances | Preserved in their existing normalized `[0, 1]` representation |
| 18–19 | Previous throttle and torque actions | Preserved in `[-1, 1]` |
| 20–21 | Measured acceleration in body coordinates | Each component divided by 16 m/s² and clipped to `[-1, 1]` |
| 22 | Remaining energy fraction | Energy divided by fixed 100 J and clipped to `[0, 1]` |

The energy denominator is always 100 J. It is not derived from
`PhysicsConfig.max_energy`, configured initial energy, or an agent's
starting energy.

The lidar channels are not rescaled by the observation assembler.
They preserve the normalized distances returned by `sense_lidar`,
including the sensor's configured range behavior.

### Completed-transition acceleration

Acceleration is supplied by the caller as the measured finite
difference across the completed transition:

`(v_t - v_(t-1)) / dt`

It is not substituted with a P0 diagnostic or an instantaneous
pre-transition value.

The supplied acceleration is expressed in world coordinates and
rotated into the body frame using the current, post-transition
heading. Reset values of zero represent the no-history initialization
sentinel.

### Active and dead-agent semantics

`physics.alive` is the sole active mask for observation assembly.

For alive rows, the observation preserves the defined channels,
including lidar and valid zero-energy rows. A row with zero energy is
not automatically treated as dead if its physics alive mask remains
true.

For dead rows:

- Channels 0–9 and 18–22 are zero.
- Channels 10–17 are eight ones, the maximum-range lidar sentinel.

This canonicalization does not depend on terminal lifecycle
annotations such as extracted or failed. Those annotations do not
replace the physics alive mask.

All lidar rows, including rows belonging to dead agents, are subject
to input validation before dead-row canonicalization. Valid normalized
lidar values for dead rows are then canonicalized to eight `1.0`
sentinels; invalid lidar input is not exempted from validation merely
because its agent is dead.

### Validation and numerical behavior

Observation assembly validates its inputs, including finite values,
compatible shapes, dtype and device consistency, and action bounds.
It does not mutate the supplied episode state or lidar input and does
not return an aliased observation.

The implementation supports CPU and CUDA execution and preserves the
input floating-point dtype and device.

Numerical regressions cover small and extreme represented values,
including:

- Adjacent representable coordinates with a nonzero displacement.
- Opposite-sign near-maximum finite coordinates without subtraction
  overflow.
- Smallest-normal displacement behavior and preservation of its
  bearing direction.
- Near-maximum finite velocity and acceleration normalization.
- Finite output for extreme valid inputs.
- Nonzero tiny target displacement not being mistaken for the
  zero-distance bearing sentinel.

The coordinate-difference implementation avoids independently
normalizing operands before subtraction, which could erase a small
represented displacement. Its numerical handling also covers
opposite-sign finite extremes without overflowing the displacement
calculation.

These tests establish the implemented floating-point contracts for
the tested dtypes and devices. They do not claim precision beyond the
information represented by the stored inputs.

## Validation results

All results below were reported from actual executions against the
exact source revision identified in this document.

### Full CPU suite

- Result: **905 passed, 131 genuine CUDA skips**
- Failures: **0**
- Warnings: **34 existing collision-fixture warnings**
- Runtime: **1.85 seconds**
- Python: **3.10.12**
- PyTorch: **2.14.1+cpu**

The 131 skips are CUDA-availability-gated tests skipped because the
CPU runtime had no CUDA device. They are not test failures.

### Full free Tesla T4 suite

- Result: **1,036 passed, 0 skipped, 0 failed**
- Warnings: **34 existing collision-fixture warnings**
- Runtime: **8.70 seconds**
- Python: **3.13.16**
- PyTorch: **2.14.1+cu130**
- CUDA: **13.0**
- GPU: **Tesla T4**
- Reported VRAM: **14.56 GiB**

The full suite ran on an actual Tesla T4-equipped runtime. Zero skips
means the suite did not skip its CUDA-availability-gated tests in that
runtime. It does not mean every test instantiated CUDA tensors or
independently exercised CUDA parity.

The 34 warnings are the existing tensor-copy warnings in
`tests/test_p13_collision.py`; they were present in the reported
acceptance runs and are not reported as new P1.6 failures.

### P0 validator on the tested T4 source commit

The P0 validator passed on the same exact source HEAD:

- Steps: **2,400**
- Duration: **40.0 seconds**
- Validator return code: **0**

Reported maximum absolute deviations:

| Field | Maximum absolute deviation |
| --- | ---: |
| Position | 0.0001220703125 |
| Velocity | 0.000003814697265625 |
| Angle | 0.0 |
| Angular velocity | 0.0 |
| Energy | 0.0 |

These long-rollout differences are observational results. They do not
introduce a new cross-device tolerance gate or change the frozen P0
contract.

### Source CI

CI for the exact P1.6 source commit completed successfully:

- [Source CI run 45034642956](https://github.com/manav4u/LAWLESS/actions/runs/45034642956)
- Tested source commit:
  `8a0b181c2d0aa7e93ed455fb783160b8db2f92cf`
- Result: completed successfully

This is the source-commit CI result. It does not assert CI success
for a subsequent documentation-only commit or a later main-branch
integration.

## Raw execution evidence

The evidence directory contains the project-only output logs copied
from the actual validation executions:

- `cpu_pytest.txt`
- `t4_validation.txt`

`cpu_pytest.txt` records the full CPU pytest output, including the
905 passed, 131 skipped, and 34 warnings summary.

`t4_validation.txt` records the exact validated HEAD, runtime
identification, full pytest output, and P0 validator output, including
the 1,036 passed result, zero skips, zero failures, 34 warnings, and
successful validator return codes.

The validation ran on an ordinary free Google Colab Tesla T4 allocation.
No paid compute was selected or allocated. The Colab session was cleaned
up after validation.

## Scope boundary and next milestone

P1.6 establishes the 23-channel observation assembly contract. It does
not implement controllers, optimization, training, or evolution.

P1.7 remains the next milestone. P1.8 and later controller,
optimization, training, and evolution work remain incomplete.

The frozen P0 tag remains at:

`d82d8e1d910285b4d95ca7a64ad71d4a2437a883`

This P1.6 acceptance does not move or replace that tag, claim
whole-project completion, or authorize a P6 research release. Any
research release remains subject to the project owner's preview and
approval.
