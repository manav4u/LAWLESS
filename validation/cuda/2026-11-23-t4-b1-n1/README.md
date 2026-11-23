# LAWLESS CUDA B1/N1 Diagnostic Checkpoint

Date: 2026-11-23

## Scope

Evidence-only checkpoint of the B1/N1 CUDA correctness episode,
the earlier deadline-failed pilot, CPU-only supervisor validation,
and the assembled but unexecuted component probe.

No production changes, optimization, P1.8 acceptance,
or transfer audit are claimed.

## Source and configuration

- Source checkout: 0a6b409e9433a428cb6b34e02f0cd96c7cb85b6a
- Seed: 7
- Dtype: float32
- Configuration: B1/N1
- Expected time-limit truncation: 1
- No extraction performed.

## B1/N1 correctness episode

- 2,400 transitions
- 40 simulated seconds
- Initial/final audits: 21 total reachable tensors, all on CUDA.
- Floating tensors: finite at both audit boundaries.
- Non-floating tensor finiteness: not applicable.
- Instrumented episode-region elapsed:
  302.121484686 seconds.
- elapsed_loop_seconds:
  302.120304718 seconds.

These timings are instrumented episode-region and loop measurements,
respectively. Neither is an uninstrumented performance baseline.

Raw episode evidence is preserved in the original bundle.
Independent source checks are retained with the evidence and
receipts; this README does not replace them.

## Earlier deadline-failed pilot

- Stopped at the child deadline.
- 2,097 of 2,400 transitions completed.
- No final audit.
- Not a correctness pass.

Its original archive and receipt are retained separately.

## CPU-only supervisor validation

The persistent CPU-only fake-process harness passed the six cases:

- Success.
- Exit status 7.
- Hang/deadline.
- TERM-ignoring parent and grandchild.
- Leader exit with surviving descendant.
- Missing probe.

Invalid timeout inputs, own-process-group refusal, archive byte
count, SHA256, Base64 reconstruction and process cleanup checks
also passed.

The supervisor and harness are preserved under their exact
filenames.

Run the persistent CPU-only harness from this checkpoint directory:

python validate_lawless_cuda_supervisor.py

This runs fake-process tests only. It does not execute the
component probe or allocate GPU memory.

## Supervisor limitations

Linux /proc and process-group semantics are required.

The supervisor applies a 150-second execution deadline and bounded
cleanup, followed by recovery packaging. Process-group cleanup is
verified, but process-group supervision is not a universal
containment guarantee. Failure to verify cleanup or complete
packaging must remain a failure, not a successful run.

## Component probe

lawless_cuda_component_probe_final.py is assembled and unexecuted.

It must be run through the validated supervisor only when a fresh
capacity signal is available. There is no automatic retry or
automatic execution. This checkpoint itself does not authorize
GPU allocation.

## Evidence

Original raw archives, receipts, independent checks and the
compact report are preserved under evidence/.

SHA256SUMS records the delivered artifact hashes.

## Repository checkpoint

This validation checkpoint advances the validation branch from
parent 3bb5d9f.

Main at ac7e047 and the frozen P0-validated tag remain unchanged.

No P1.8 acceptance, B16/N256 full episode, uninstrumented
benchmark or transfer audit is claimed.
