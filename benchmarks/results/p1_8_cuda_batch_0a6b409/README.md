
# P1.8 bounded CUDA batch-equivalence evidence

**Status: PASS — bounded test only; P1.8 remains open.**

- Pinned checkout: `0a6b409e9433a428cb6b34e02f0cd96c7cb85b6a`
- Hardware/runtime: free Colab Tesla T4; Python 3.13.16; PyTorch 2.14.1+cu130; CUDA 13.0.
- Test: `tests/test_p1_cuda_batch_equivalence.py`
- Test bytes: `16,533`
- Test SHA-256: `b36fcadd7232faf6eba4ac7549924aba1a53deb2c9f27085eebbe4a1fe542975`
- Result: **2 passed, 0 skipped**, pytest exit 0, 5.44 seconds.

## Coverage

1. **Batched world and candidate equivalence:** B=2, N=2, float32; five public `step()` transitions after CUDA reset. Compares batched worlds against independent one-world executions and batched candidates against independent one-candidate references. Exercises the configured gravity shift.
2. **Mixed completion and freeze:** world 0 physically completes through low-energy death (`1e-6` fixture); world 1 remains active and reaches deadline truncation. Checks recursive frozen-world snapshots, observations and terminal masks, zero per-transition events for the frozen world, active-world progression, and recursive immutability after rejected all-complete stepping.

Numerical comparison: `rtol=1e-5`, `atol=1e-6`; discrete values exact. Non-finite floating-point values are rejected.

## Evidence integrity

- Recovered final ZIP: **5,391 bytes**
- ZIP SHA-256: `4aa2eb17b006a409279668ac20dac28239995859c97c9379f2f79ed19c13f94b`
- Exact test size and SHA-256 verified.
- All eight production-source/helper hashes matched the local reference.
- Pinned checkout remained clean and unchanged before and after execution.

The ZIP's internal `evidence_zip` metadata (`5,280` bytes and its first-build hash) is stale because the receipt was rewritten during packaging. **Use the recovered final ZIP's 5,391-byte size and SHA-256 above as authoritative.**

## Existing CPU suite

Full local console suite: **1,173 passed, 137 genuine CUDA-unavailable skips, 34 warnings**, 114.20 seconds, explicit exit 0. CUDA skips are expected in the CPU environment and are separate from the successful T4 run.

## Scope and next gate

This is bounded B=2/N=2 correctness evidence, not full P1.8 acceptance, a performance result, or phase closure. Preserve production `main`, P0, A2.1, and all four A2.2 files unchanged.

Next original P1.8 gate: representative **B=16, N=256** GPU memory/shape evidence using existing production code, with no optimization or production edits. Record actual allocation/peak-memory and shape results; do not infer performance acceptance from this correctness test.
