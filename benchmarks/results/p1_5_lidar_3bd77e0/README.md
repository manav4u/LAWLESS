# P1.5 Eight-Ray Lidar Validation

## Status

P1.5 eight-ray lidar source validation is complete at exact source
commit `3bd77e057e46f8026abd422baf034332261975bf` on `validation/lidar`.

The exact tested source revision passed the focused CPU suite
(97 passed, 63 genuine CUDA skips), full CPU suite
(814 passed, 119 genuine CUDA skips, zero failures), and full free
Tesla T4 suite (933 passed, zero skips, zero failures). The P0
validator passed 2,400 steps over 40 seconds on both CPU and T4.
The full-suite runs each reported 34 unchanged prior collision-fixture
warnings.

A documentation-only successor,
`a8ae378be04cea981819f82d4f7610d5f603baa9`, was subsequently pushed
to `validation/lidar` and fast-forwarded to `main`. Its branch CI and
the CI for the integrated `main` both passed, as recorded below.
The successor changes documentation only; its product source, tests,
and validator remain identical to the T4-tested P1.5 source commit.

This evidence records P1.5 validation and its completed integration.
It does not claim overall project completion or authorize a later
research release.

## Scope and source provenance

- Exact tested source: `3bd77e057e46f8026abd422baf034332261975bf`
- Source branch: `validation/lidar`
- Accepted P1.4 base on `main`: `994dc95280a582d56df651c6741a2787595d529a`
- P1.4 validated source: `f03704d58500de041bbfad7403bb315878110052`
- Frozen P0 tag: `p0-validated`
- Frozen P0 commit: `d82d8e1d910285b4d95ca7a64ad71d4a2437a883`

The P1.5 source commit adds only:

- `lawless/sensors/__init__.py`
- `lawless/sensors/lidar.py`
- `tests/test_p15_lidar.py`

The complete existing test suite and all previously accepted source
modules were retained. P1.5 tests are append-only. No product, test,
or validator changes were made after the tested source commit.

## Sensor contract

The sensor returns eight normalized distances in `[0, 1]`, ordered at
body-relative angular offsets `2*pi*i/8`, for `i = 0..7`.

- Rays originate at the agent center.
- The default maximum range is 20 metres and is configurable through
  `SensorConfig.lidar_range_m`.
- Ray directions are rotated by the stored agent heading.
- The nearest forward obstacle-circle surface or rectangular world
  boundary is reported.
- A hit at or beyond maximum range is represented by `1.0`.
- No hit within range is represented by `1.0`.
- Dead agents receive eight `1.0` values.
- A live agent outside the world is rejected.
- A dead agent may be outside the world without invalidating other
  live agents in the same batch.
- Sensing is read-only and does not mutate agent, world, or obstacle state.

Obstacle sensing uses circle surfaces, not agent-radius inflation.
Cores, extraction, collision resolution, and observation assembly are
outside this sensor's scope.

## Numerical and dtype policy

Agent and obstacle geometry inputs must use `torch.float32` or
`torch.float64`. Other geometry dtypes, including `torch.float16`, are
rejected.

Geometric calculations are promoted to `torch.float64` on the input
device. The returned tensor retains the agent position's original dtype
and device. Promotion improves arithmetic stability but cannot recover
precision already lost when values were stored in float32.

The eight base ray directions use an exact body-relative direction
table. The stored heading is applied without an angular dead zone or
small-angle snapping. Direction vectors are normalized after rotation.

Circle tangency is inclusive under the computed float64
perpendicular-distance comparison. No epsilon halo, radius inflation,
or near-contact tolerance is used.

Exact represented surface contact is handled independently of ray
direction: when the represented squared center distance is exactly
equal to the represented radius squared, every ray reports zero.
Strictly interior origins use the forward exit surface. Strictly
exterior origins use the first forward intersection when one exists.

These rules distinguish exact represented contact from adjacent
representable interior and exterior positions. They do not claim to
recover an ideal real-number geometry beyond the precision of the
stored inputs.

## Batch and state semantics

The sensor supports scalar raw `AgentState` layouts as well as
multidimensional batches. A raw scalar `AgentState` has position shape
`(2,)` and scalar fields with shape `(1,)`.

This lower-level `AgentState` support does not change the accepted
composed `AgentEpisodeState` contract. Composed episode initialization
and validation continue to require at least one positive batch
dimension.

For batched sensing, each agent's alive status and world-membership
check are evaluated independently. Dead agents receive maximum-range
sentinels even if their stored positions are outside the world.

## Validation results

All results below were reported from actual executions against the
exact source revision identified above.

### Focused CPU tests

- Result: **97 passed, 63 genuine CUDA skips**
- No test failures.

### Full CPU suite

- Result: **814 passed, 119 genuine CUDA skips**
- Failures: **0**
- Warnings: **34 unchanged prior collision-fixture warnings**
- Runtime: **1.67 seconds**
- Runtime environment: Python 3.10.12, PyTorch 2.14.1+cpu

The CUDA skips are expected when CUDA is unavailable in the CPU
runtime.

### Full free Tesla T4 suite

- Result: **933 passed, 0 skipped, 0 failed**
- Warnings: **34 unchanged prior collision-fixture warnings**
- Runtime: **9.59 seconds**
- Runtime environment:
  - Python 3.13.16
  - PyTorch 2.14.1+cu130
  - CUDA 13.0
  - Tesla T4
  - 14.56 GiB reported VRAM

This was a full-suite execution on a Tesla T4-equipped runtime.
Zero skips means the suite did not skip its CUDA-availability-gated
tests in that runtime. It does not mean every test instantiated CUDA
tensors or independently exercised CUDA parity.

### P0 validator

The P0 validator passed on CPU and on the actual Tesla T4 runtime:

- 2,400 steps
- 40.0 seconds

The T4 long-rollout maximum absolute deviations reported were:

| Field | Maximum absolute deviation |
| --- | ---: |
| Position | 0.0001220703125 |
| Velocity | 0.000003814697265625 |
| Angle | 0.0 |
| Angular velocity | 0.0 |
| Energy | 0.0 |

These long-rollout differences are observational results. They are not
a newly introduced tolerance gate or a change to the frozen P0
contract.

### Source CI

The CI run for the exact P1.5 source revision completed successfully:

- [Source CI run 45028568460](https://github.com/manav4u/LAWLESS/actions/runs/45028568460)
- Pytest job 137226880572: completed successfully.

The P1.5 source commit `3bd77e057e46f8026abd422baf034332261975bf`
passed source CI ([run 45028568460](https://github.com/manav4u/LAWLESS/actions/runs/45028568460)).
After the documentation-only successor
`a8ae378be04cea981819f82d4f7610d5f603baa9` was pushed to
`validation/lidar`, its CI completed successfully
([run 45030148306](https://github.com/manav4u/LAWLESS/actions/runs/45030148306)).
Following the fast-forward of `main` to that same exact revision,
integrated-main CI also completed successfully
([run 45030366342](https://github.com/manav4u/LAWLESS/actions/runs/45030366342);
[pytest job 137232422975](https://github.com/manav4u/LAWLESS/actions/runs/45030366342/job/137232422975)).
These are historical CI results for their respective exact commits.
They do not assert CI success for any subsequent documentation-only
receipt commit.

## Raw execution evidence

The following files were copied unchanged from the actual validation
runs:

- `focused_cpu_pytest.txt`
- `full_cpu_pytest.txt`
- `p0_cpu_validator.txt`
- `full_t4_validation.txt`

The combined T4 file contains runtime identification, full pytest
output, and P0 validator output, including zero return codes for both
executed commands.

The full T4 run was performed using a free Colab Tesla T4 runtime.
The free account and zero compute-unit balance were checked before
allocation. After validation, the GPU runtime was disconnected and
deleted; the session manager showed no active session. No paid
compute was used.

## Regression coverage retained

The P1.5 test suite retains the original sensor tests and the
append-only regression tests, including:

- Output shape, dtype, device, normalization, and ray ordering.
- Heading rotation and nearest obstacle-versus-boundary selection.
- Empty obstacle worlds and boundary sensing.
- Inclusive tangency, strict near-tangent misses, and obstacles behind
  or beyond range.
- Interior-origin forward exits and exact surface-origin zero results.
- Edge and corner behavior, including outward and inward rays.
- Live-outside rejection and dead-agent maximum-range sentinels.
- Raw scalar `AgentState` support and independent scalar/batch parity.
- Multidimensional and non-square batches.
- Sensor configuration and invalid-input validation.
- Mutable obstacle geometry revalidation.
- Read-only sensing.
- Genuine CUDA parity and CUDA-specific numerical regressions.
- Small nonzero headings at edges and corners.
- Oblique strict near-tangent misses.
- Heterogeneous obstacle radii and nearest-surface selection across
  both obstacle orders.
- Custom range normalization and clipping at 5, 10, and 30 metres.
- Exact 3-4-5 surface contact for every ray.
- Adjacent representable interior and exterior surface positions.

The reported focused and full test results include these retained
regressions.

## Known limitations and deferred work

P1.5 does not add swept collision detection. The previously documented
P1.3 collision resolver provides discrete endpoint contact resolution,
not a guarantee against tunneling between endpoints.

P1.7 will later consume boundary and failure reports as part of episode
integration. That report-consumption work is not part of P1.5.

The sensor does not assemble the final 23-dimensional observation.
That belongs to P1.6. Episode integration, including the planned
20-second gravity shift and 40-second episode, belongs to P1.7.
Vectorization, parity, and performance work belongs to P1.8.

The following stages remain incomplete:

- P1.6: exact 23-dimensional observation assembly.
- P1.7: episode integration and report consumption.
- P1.8: vectorization, parity, and performance work.
- Controllers: static, recurrent, and plastic controllers.
- Optimization and evolution: GA, OpenAI-ES, and CMA-ES comparison.
- Training and later research phases.

No broad performance claim or overall project-completion claim is
made by this validation.

## Integration and release boundary

The documentation-only successor
`a8ae378be04cea981819f82d4f7610d5f603baa9` was pushed to
`validation/lidar`; its CI completed successfully. `main` was then
fast-forwarded to the same exact commit, and integrated-main CI,
including its pytest job, completed successfully. The live branch
refs were verified at that hash.

The successor contains documentation and evidence only. Its product
source, tests, and validator are identical to the P1.5 source revision
`3bd77e057e46f8026abd422baf034332261975bf` validated on the Tesla T4.
No additional product, test, or validator changes are represented.

The frozen P0 tag remains unchanged at
`d82d8e1d910285b4d95ca7a64ad71d4a2437a883`.

No GPU session remains active, and credentials were cleaned after
validation. This integration does not authorize paid compute or a
P6 research release. Any later release remains subject to the
project owner's preview and approval. P1.6-P1.8, controllers,
optimization, training, and evolution remain incomplete.
