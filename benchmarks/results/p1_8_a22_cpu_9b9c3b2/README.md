# A2.2 CPU profiling and timing evidence

This directory archives the A2.2 CPU component-profile and uninstrumented timing evidence captured from **one GitHub Actions hosted CPU run**. The profile receipt, timing receipt, original run manifest, and inner evidence ZIP all belong to that same execution. They are not a separate Colab execution and must not be described as Colab results.

## Run provenance

- Repository: https://github.com/manav4u/LAWLESS
- Validation branch: `validation/environment-benchmarks`
- Source checkpoint: `9b9c3b28a41037671b0806d234e1d728082a9b28`
- Automation commit: `e832f2398444b2dc38e745c20618177088a51a46`
- GitHub Actions run: [45066856729](https://github.com/manav4u/LAWLESS/actions/runs/45066856729)
- GitHub Actions job: [137344837156](https://github.com/manav4u/LAWLESS/actions/runs/45066856729/job/137344837156)
- Outer Actions artifact ID: `14772171030`
- Outer Actions artifact SHA-256: `92ff23ea2e210cd0c4486fe6d03ada42c00d97fa9faec0a9aa4a5ec78457be4e`
- Supervisor status: `cell_completed`; exit code `0`; no error; `timed_out: false`
- Run-manifest overall status: `valid`

The outer Actions artifact is the uploaded workflow artifact. Inside the captured evidence inventory is `LAWLESS_A22_CPU_evidence.zip`, the **inner evidence ZIP**, SHA-256 `19f9e8cf58c619355804626bf63383a2721ddf81011a1f3ead41f4b6133389f0`, size 28,070 bytes. These are two different archive layers from the same Actions run: do not conflate their hashes or describe them as separate executions.

The operational cell retains the legacy filename `scripts/lawless_p18_a22_colab_cell_candidate.py` and its legacy Colab-oriented schema/workspace naming. Those names were reused by the hosted CPU automation and do not identify the execution environment. The expected and actual operational-cell SHA-256 was `5c34a9f3fc7d4d0f8463a3dbc70e37544776fd4d44aecfe18f542b53231bee51`. The supervisor's runtime-cell SHA-256 after its observability patches was `2b25fb888a03828c40dae907c08964be5c7e5a2fe577fc966c5c8498122406d8`.

An earlier interactive Colab attempt disconnected and produced no recovered receipt. It is not the source of any artifact archived here. All four raw files listed below were captured by the supervisor in the GitHub Actions run identified above.

## Raw evidence files

All raw files must be copied into this directory byte-for-byte. Do not rewrite, normalize, regenerate, or edit their contents.

| File | Role | SHA-256 | Size |
|---|---|---|---:|
| `a22_cpu_component_profile.json` | Component-profile receipt | `118940bab131c73bc1e529d361ae7e1de8e4a38e3ab00408dd319b471001d61b` | 54,963 bytes |
| `a22_cpu_uninstrumented_timing.json` | Uninstrumented timing receipt | `e38f5260839373410dbd31aee4684a0007f08b24bc96045c223427b161a23cd3` | 348,815 bytes |
| `run_manifest.json` | Original run manifest, retained unchanged | `ff5ce71a05b1828d1032086cb2654cce0f00c298e711335c19af9d90948004c1` | 4,105 bytes |
| `LAWLESS_A22_CPU_evidence.zip` | Inner supervisor-captured evidence ZIP | `19f9e8cf58c619355804626bf63383a2721ddf81011a1f3ead41f4b6133389f0` | 28,070 bytes |

`manifest.json` is the archival index authored for this directory. It does not replace or modify the original `run_manifest.json`.

## Workload and runtime

| Parameter | Recorded value |
|---|---|
| Device | CPU |
| Dtype | `torch.float32` |
| Seed | `7` |
| Batch | `1` |
| Candidates | `1` |
| Episode length | `2400` steps |
| Component-profile episodes | `1` |
| Full timing warmup episodes | `1` |
| Full measured timing episodes | `5` |

Runtime recorded by the receipts:

- Linux `7.0.0-1020-azure`, x86_64
- CPython `3.11.18`
- PyTorch `2.15.0+cu132`
- Reported CUDA version: `13.2`; CUDA unavailable, execution device CPU
- PyTorch intra-op threads: `2`
- PyTorch inter-op threads: `4`
- Logical CPU count before execution: `4`
- `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` were unset

The run-manifest runtime capture began at `2026-11-22T20:32:25.283221+00:00` and its final runtime capture was `2026-11-22T20:36:48.434883+00:00`. The supervisor started at `2026-11-22T20:32:23.018346+00:00` and finished at `2026-11-22T20:36:48.863514+00:00`.

## Component profile

The profile receipt records one instrumented full episode. Component intervals are inclusive, nested wall-clock wrapper intervals. They must not be summed as independent costs.

| Component | Inclusive interval (seconds) |
|---|---:|
| Assemble observations | `1.476805489` |
| Physics | `0.348655333` |
| Resolve static obstacles | `28.054202243` |
| Sense lidar | `1.056850940999999` |

The component measurements report valid timing and 2,400 attempted/successful calls for each measured component. The profile also records:

- Controller and public-step loop: `37.445222613` seconds
- Inclusive public-step interval: `37.15986847199994` seconds
- Public-step calls attempted and returned: `2400` and `2400`
- Reset: `0.007072278` seconds
- Timing method: `perf_counter_ns`

Wrapper dispatch and clock-read overhead are included and were not separately quantified or subtracted. The component intervals are inclusive and nested; do not add them together. This is an instrumented profile, not an uninstrumented baseline.

The profile receipt contains `source_before` and `source_after` identity records. Their full equality was checked as a derived validation of unchanged source identity across profiling. This is a validation conclusion from comparing those receipt fields, not a `profile_source_fingerprint_before` or `profile_source_fingerprint_after` field: the profile receipt does not contain fields with those names. The source fingerprint values recorded in the timing evidence belong to the timing receipt and must not be attributed to the profile receipt.

The profile reports restoration status `verified`, zero restoration errors, and successful callable-identity checks for `a21_step`, `assemble_observations`, `module_step`, `physics_step`, `resolve_static_obstacles`, and `sense_lidar`.

## Uninstrumented timing

The timing receipt records one complete warmup episode followed by five complete measured episodes. The warmup duration was `37.28541197100003` seconds. Each measured episode contains 2,400 public step calls and 2,400 world steps.

| Measured repetition | Duration (seconds) |
|---:|---:|
| 1 | `37.481794820000005` |
| 2 | `37.45022661600001` |
| 3 | `37.65125348799984` |
| 4 | `37.479102294000086` |
| 5 | `37.56692981499987` |

Recorded summary:

- Median measured episode duration: `37.481794820000005` seconds
- Spread (maximum minus minimum): `0.20102687199982938` seconds
- Median rate: `64.03108526487578` world steps per second

The rate describes only this B1/N1, seed-7, CPU float32 workload and this recorded runtime. It is not a throughput or scaling claim for larger batches, different candidate counts, other hardware, or other workloads.

## Recorded validation observations

The profile and timing receipts report valid status. Recorded checks include:

- Each measured timing episode attempted 2,400 transitions, returned from 2,400 public step calls, and completed 2,400 world steps.
- Observation shape `[1, 1, 23]`; CPU float32; finite observations.
- Physical state finite.
- Final gravity `[[-0.0, 4.0]]`.
- Final inventory: extracted `[0]`, failed `[0]`, physical-alive `[1]`, truncated `[1]`.
- Geometry unchanged. Recorded geometry: body radius `0.5`, core count `8`, extraction radius `3`, obstacle count `20`, spawn count `1`.
- The timing receipt records source fingerprint `e254c052685a40b387fc00df1952a77d0c2dff0c5b0f8317b50a40746c58f972` before and after timing, with matching values.
- Source identity records report 79 tracked Python files, zero untracked Python files, a clean working tree, and frozen P0 comparison `20/20` preserved.

These are the checks recorded by the supplied receipts. The profile source-before/source-after equality statement above is a derived comparison of the full identity records; it is not a claim that the profile receipt has timing-receipt fingerprint fields.

## Historical declared-source field

The receipts retain the A2.1 benchmark's historical declaration:

`DECLARED_SOURCE_REVISION=ac7e0471ae314460d6f72036a33f9a2e3a198bc5`

The executed checkout revision recorded by the run is:

`9b9c3b28a41037671b0806d234e1d728082a9b28`

Accordingly, `declared_revision_matches_checkout` is `false`. Preserve this declaration and mismatch exactly as recorded; do not rewrite the field to `true` or omit it.

The supplied A2.1 `source_identity()` implementation returns this comparison as an informational field and does not use it to determine `identity_status`. The recorded identity checks instead include runtime revision, tracked and untracked Python inventories, file hashes, clean-tree status, and frozen P0 comparison. The receipts report the relevant source identity checks as verified. This interpretation does not mean that the historical declaration matches the executed checkout.

## Scope and limitations

This archive documents CPU profiling and timing for the stated B1/N1, seed-7 workload only. It does not establish larger-batch performance, batch equivalence, memory feasibility at larger workload sizes, GPU/T4 performance, scaling, or a production performance improvement. The component profile is instrumented and must not be represented as an uninstrumented baseline or as an additive breakdown of independent costs.

The evidence does not claim a no-transfer audit, P1.8-A acceptance, a main merge, or P6 release readiness. No production, A2.1, or frozen P0 source edits are claimed by this run. The automation and run provenance recorded here do not authorize broader acceptance or release conclusions.

## Integrity verification

From this directory, verify the four archived raw files using the following checksum list:

```text
118940bab131c73bc1e529d361ae7e1de8e4a38e3ab00408dd319b471001d61b  a22_cpu_component_profile.json
e38f5260839373410dbd31aee4684a0007f08b24bc96045c223427b161a23cd3  a22_cpu_uninstrumented_timing.json
ff5ce71a05b1828d1032086cb2654cce0f00c298e711335c19af9d90948004c1  run_manifest.json
19f9e8cf58c619355804626bf63383a2721ddf81011a1f3ead41f4b6133389f0  LAWLESS_A22_CPU_evidence.zip
```

Save the lines above as `SHA256SUMS` if a standalone checksum file is desired, then run:

```sh
sha256sum -c SHA256SUMS
```

The outer Actions artifact hash is `92ff23ea2e210cd0c4486fe6d03ada42c00d97fa9faec0a9aa4a5ec78457be4e`; it identifies the uploaded Actions artifact, not the inner `LAWLESS_A22_CPU_evidence.zip` file. The outer artifact is not one of the four files listed in the inner captured-file inventory.
