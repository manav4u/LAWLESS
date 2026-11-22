# P1.3 Static-Circle Collision Validation

## Status and source provenance

P1.3 static-contact resolution passed CPU and free Tesla T4 validation.

- **Validated source snapshot:** `e11e9e669cad23716336a65ec19f1ab1d60a6a5e`
- **Original identical product/test source commit:** `e7e628203095c3a4b8ce091f1e27e9f8db343c5a`
- **Branch:** `validation/collisions`
- The validated snapshot `e11e9e6` adds only the candidate README status relative to `e7e6282`. The product, tests, and validator source are identical.
- **Pre-integration main baseline:** `30a3fc467a27db7babe134618925f55ea68ac297` (P1.2)
- **Frozen P0 tag:** `p0-validated`, peeling to `d82d8e1d910285b4d95ca7a64ad71d4a2437a883`
- **Physics diff against frozen P0:** empty

The full CPU and T4 validation runs were performed at the exact snapshot `e11e9e669cad23716336a65ec19f1ab1d60a6a5e`.

## Test results

| Validation | Result |
|---|---|
| Full CPU suite | 598 passed, 52 expected hardware-dependent CUDA skips, 0 failures |
| Full Tesla T4 suite | 650 passed, 0 skips, 0 failures |
| T4 full-suite runtime | 6.67 seconds |
| P0 CPU validator | 2400 steps / 40 seconds , passed |
| P0 T4 validator | 2400 steps / 40 seconds , passed |
| P0 CPU/T4 alive masks | Equal |

## Environment

- Python `3.13.16`
- PyTorch `2.14.1+cu130`
- CUDA `13.0`
- NVIDIA Tesla T4, `14.56 GiB`
- Free T4 allocation; no paid compute or upgrade

## P0 CPU/T4 comparison

Maximum absolute CPU-to-T4 deviations:

| Field | Maximum absolute deviation |
|---|---:|
| Position | `0.0001220703125` |
| Velocity | `0.000003814697265625` |
| Angle | `0` |
| Angular velocity | `0` |
| Energy | `0` |

These are observed rollout differences, not a newly imposed tolerance or acceptance gate. Alive masks matched.

## CI

Candidate/source CI passed:

[GitHub Actions run 45018527407](https://github.com/manav4u/LAWLESS/actions/runs/45018527407)

CI for any later documentation-only integration commit is tracked separately from this candidate/source CI result.

## Raw evidence files

The following raw files are to accompany this README in the integration:

- `cpu_pytest.txt` , full CPU test output
- `t4_pytest.txt` , full Tesla T4 test output
- `p0_cpu.txt` , P0 CPU validator output
- `p0_t4.txt` , P0 T4 validator output
- `runtime.txt` , validation environment and runtime details

## Implemented P1.3 scope

- Deterministic, ordered, discrete endpoint resolution against static circles
- Positional depenetration and zero-restitution removal of inward normal velocity
- Preservation of tangential velocity
- Per-agent scale-aware tolerance
- Deterministic coincident-center fallback
- Default and enforced maximum of eight solver passes
- Separate `contact`, `unresolved_contact`, and `boundary_escape` reports, each shaped `batch_shape + (1,)`
- Dead-agent freezing and exclusion from collision response
- Input immutability
- Real episode composition coverage for objective, history, and lifecycle preservation
- Finite-value and parameter validation

## Limitations and future scope

This is a discrete endpoint-contact solver, not swept or continuous collision detection. No guarantee against tunneling between endpoints is made. The observed test runtime is not a general performance guarantee.

P1.3 reports boundary escape and unresolved contact; it does not consume those reports in episode stepping. P1.7 is responsible for later boundary/failure consumption.

Eligibility/reward, full observations, episode stepping, controllers, optimization, and evolution remain future work. This evidence does not declare the overall project complete.

## Integration provenance

The tested snapshot is `e11e9e669cad23716336a65ec19f1ab1d60a6a5e`. The earlier commit `e7e628203095c3a4b8ce091f1e27e9f8db343c5a` contains the identical product/test source. The documentation-only integration adds the raw validation logs, this P1.3 evidence README, and the updated main README status paragraph. It makes no product, test, or validator changes relative to `e11e9e6`. CI for the documentation-only integration is tracked separately from candidate/source CI.
