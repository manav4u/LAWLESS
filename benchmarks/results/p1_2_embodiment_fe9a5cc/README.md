# P1.2 Embodiment and Lifecycle Validation

**Validated candidate:** `fe9a5cca6c57bbd3edca1ec1e375804686cd8a0a`

**Status:** CPU, CUDA/T4, frozen P0 validation, and CI passed. This evidence refers to the validated source candidate, not a later documentation-only commit.

## Scope

P1.2 composes independent per-agent embodiment state, core-collection bookkeeping, explicit spawn assignments, reset, clone, and lifecycle state around the existing P0 physics state and shared generated-world layout.

The implementation preserves the frozen P0 physics semantics and composes per-agent state without placing mutable episode bookkeeping into the shared generated-world layout.

Implemented and validated composition includes:

- Embodiment radius consistency with the generated layout.
- Per-agent physical and embodiment state.
- Alive state and terminal failure/extraction bookkeeping.
- Independent per-agent core-collection state.
- Per-agent episode state and explicit spawn-index mapping.
- Previous-action storage and explicit measured-acceleration provenance.
- Batch-shape validation and placement consistency.
- Reset and clone storage isolation.
- Existing frozen-dead-state and no-resurrection protections.
- Shared generated-world layout identity and read-only usage.

An acceleration buffer initialized to zero at reset is a placeholder: it does not represent a prior measured acceleration. A measurement is recorded only through an explicit transition-measurement recording operation. No separate measurement-availability flag is claimed.

## Validation results

| Gate | Result |
|---|---|
| Full CPU suite, exact candidate | 547 passed, 50 expected CUDA skips |
| Full CUDA/T4 suite, exact candidate | 597 passed, 0 skips, 0 failed |
| Frozen P0 CPU validator | 2400 steps / 40 seconds; passed |
| Frozen P0 T4 validator | 2400 steps / 40 seconds; passed |
| CI for exact candidate | Passed |
| Physics diff against frozen P0 | Empty |
| Frozen P0 tag | Unchanged: `d82d8e1` |

### CPU suite

- 547 passed.
- 50 expected CUDA skips.
- Runtime: 1.09 seconds.
- Raw log: [cpu_pytest.txt](cpu_pytest.txt)

### CUDA/T4 suite

- 597 passed.
- 0 skips.
- 0 failures.
- Runtime: 2.81 seconds.
- Raw log: [t4_pytest.txt](t4_pytest.txt)

The actual free T4 validation completed successfully. The Colab runtime was subsequently deleted, and the session manager confirmed no active sessions. The recorded results and raw logs are retained as validation evidence; this README does not imply a currently active T4 runtime.

### Frozen P0 validator

The frozen P0 2400-step / 40-second validator passed on both CPU and T4.

Checks included:

- Per-step finite-value checks.
- Energy checks.
- Angle checks.
- CPU/GPU alive-state equality.

Raw logs:

- [p0_cpu.txt](p0_cpu.txt)
- [p0_t4.txt](p0_t4.txt)

### Reported T4 maximum absolute deviations

| Quantity | Maximum absolute deviation |
|---|---:|
| Position | `0.0001220703125` |
| Velocity | `0.000003814697265625` |
| Angle | `0` |
| Angular velocity | `0` |
| Energy | `0` |

These values are the reported frozen-P0 CPU/T4 comparison results, not new benchmark measurements.

## CI

The exact candidate CI run passed:

[GitHub Actions run 45015939959](https://github.com/manav4u/LAWLESS/actions/runs/45015939959)

## Runtime environment

Recorded T4 validation environment:

- Python: `3.13.16`
- PyTorch: `2.14.1+cu130`
- CUDA: `13.0`
- GPU: Tesla T4
- GPU memory: 14.56 GiB

Raw environment log: [runtime.txt](runtime.txt)

## Candidate integrity and baseline protection

Validated source commit:

`fe9a5cca6c57bbd3edca1ec1e375804686cd8a0a`

The product-code diff against the preceding candidate `6c71705` is empty. The final correction changed only device assertions in tests, comparing exact device placement against the expected tensor/layout device rather than an unindexed CUDA device request.

The physics diff against `p0-validated` is empty. The frozen P0 tag remains unchanged at `d82d8e1`.

The validated candidate is the source of these results. Later documentation or integration commits do not change the source commit against which the validation was performed.

## P1.2 acceptance boundary

P1.2 is embodiment and lifecycle composition, not a collision or episode-execution implementation.

Still unimplemented:

- P1.3 collision/contact response.
- P1.4 contact/pickup/extraction eligibility and reward integration.
- P1.6 observation integration.
- P1.7 episode stepping.

Controller comparisons, optimization, training, and evolution remain future work.

P0 remains validated and frozen.
