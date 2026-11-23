## LAWLESS P1.8 — CUDA evidence checkpoint (2026-11-23)

### Bounded CUDA equivalence — PASS

- Exact test: 9,068 bytes.
- Test SHA-256: `522f0662d3703f3f01ac4fe48073213478a45645619f50bb3fc51654a3a9d096`
- Pinned source: `0a6b409e9433a428cb6b34e02f0cd96c7cb85b6a`
- Runtime: free Colab Tesla T4, Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0.
- Pytest: **exit 0; 1 passed in 5.67s; zero skips.**
- Checkout unchanged and clean.
- Scope: B=2, N=2, seed=710, float32; reset plus three public steps; gravity-shift fixture; outputs and state tree compared.
- Tolerances: **rtol=1e-5, atol=1e-6**; discrete values exact.

### Raw equivalence archive

- `LAWLESS_cuda_equivalence_20261123T002206_410010Z.zip`
- Final recovered size: **3,702 bytes**
- SHA-256: `bc3e0ca978de619cb17ac1bef22b469b960e9d5bed0bf30854757bcd49e77d2f`
- Recovered Base64, ZIP integrity, packaged test size and hash verified.
- Wrapper `zip_bytes=3685` is stale first-build metadata; final recovered size/hash are authoritative. Not a test or source failure.

### Original A1 CUDA smoke — separate receipt

- ZIP size: **24,271 bytes**
- SHA-256: `94ba92bff2645f053fda56e43c3be8bf36c701b5f4540a02f9764a0c276103b8`
- Source inventory hashes verified.
- B=16, N=256; 16 worlds × 2 steps = **32 world-step transitions**.
- **4,096 initialized candidate slots**; 8,192 candidate-slot transitions across two steps.
- Bounded incomplete/max-steps execution; expected exit 2.
- Smoke evidence only; not a completed baseline or full episode.

### Existing CPU evidence retained

- Full local console suite: **1,173 passed, 135 genuine CUDA-unavailable skips, 34 warnings, 113.97s, explicit exit 0**.
- Existing bounded CPU correctness receipts remain credited: base fixture `e61d0bb0b2862c34764adb9fd8eb9079d9cf091d`; objective progression `effd11e7c1e8fe5a2d9b4cf882c425ec6fe033ad` with import fix `e0c57c2` (hosted run `45078720317` GREEN); mixed completion/world freeze `0a6b409e9433a428cb6b34e02f0cd96c7cb85b6a`.
- CPU profiling and five warmed full-episode measurements remain archived at `98dbec35`. Original deterministic CPU episode and fixed-seed repeatability coverage are credited, not reopened.

### Change boundary

Production `main`, P0, A2.1, and all four A2.2 files unchanged. Test/evidence work only; no paid compute or main merge.

### Remaining original P1.8 evidence limits

- Broader GPU correctness coverage: batched-world versus independent-world, batched-agent, and CPU/T4 observation equivalence at representative workload scales. Current CUDA equivalence is only the bounded B=2, N=2, three-step case.
- GPU memory and shape sanity, including peak-memory evidence.
- No-unintended-CPU-transfer evidence across the relevant execution path.
- Full CUDA component timing: initialization, reset, physics/collision, lidar, observations, and complete step.
- Full 40-second CUDA episode timing and CPU/T4 comparison, reporting measured agent-steps/s and episodes/s.

The original B=16, N=256 workload is an evidence workload, not a mandatory speed target. No additional CUDA deterministic-episode gate is imposed by this checkpoint.

**P1.8 is NOT ACCEPTED and full-phase completion is NOT CLAIMED; bounded CUDA equivalence, A1 smoke, existing CPU correctness/profiling evidence, and the full local regression are recorded, with the remaining GPU coverage, memory, transfer, and timing evidence still outstanding.**
