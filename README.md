# LAWLESS

**Evolution of adaptive control under non-stationary physics.**

LAWLESS investigates whether evolution can produce agents whose within-lifetime learning mechanisms improve adaptation after unexpected changes in environmental dynamics.

> Current status: **P0 — deterministic physics foundation**. No learning or intelligence claims are made by the P0 engine.

## V1 decisions

- Controllers: static feedforward, recurrent, and plastic (future milestones)
- Optimizers: genetic algorithm, OpenAI-ES, and CMA-ES (future milestones)
- Environment: continuous 2D navigation (future milestones)
- Compute: hybrid CPU + CUDA where beneficial
- P0: NumPy/PyTorch-compatible batched physics, validation, and benchmarks

## Coordinate and unit convention

Internal world coordinates are Cartesian: +X right, +Y up, positive angles counterclockwise. SI units are used internally: m, s, kg, N, rad, J. Rendering converts world coordinates to screen coordinates; pixels never enter physics calculations.

## P0 physics

For heading `theta`, forward direction is `d = (cos(theta), sin(theta))`.

`a = (F_max / m) * throttle * d + gravity - c_v * velocity`

`alpha = (tau_max / I) * torque_command - c_omega * omega`

Semi-implicit Euler:

- `v_next = v + a * dt`
- `p_next = p + v_next * dt`
- `omega_next = omega + alpha * dt`
- `theta_next = wrap(theta + omega_next * dt)`

Actions are clipped to [-1, 1]. Energy use is `dt * (P_thrust * abs(throttle) + P_torque * abs(torque_command))`; exhausted agents have actuators disabled.

## Start locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e ".[dev]"
pytest -q
python benchmarks/physics_benchmark.py --agents 100 500 1000 2500 5000 --steps 2000
```

## Google Colab

Open [`notebooks/LAWLESS_P0_Colab.ipynb`](notebooks/LAWLESS_P0_Colab.ipynb) in Google Colab. The notebook clones this repository, installs dependencies, runs tests, executes a deterministic sanity check, and benchmarks CPU/GPU backends.

After creating `manav4u/LAWLESS` on GitHub, set the notebook's `REPO_URL` to `https://github.com/manav4u/LAWLESS.git` (already configured).

## P0 Validation Acceptance

P0 physics acceptance requires:

* **Analytical dynamics:** physics integration and actuator/energy behavior satisfy the defined physics contracts and analytical tests.
* **Deterministic CPU:** repeated runs with identical initial conditions and actions reproduce identical results.
* **Batched vs. independent stepping:** batched simulation agrees with independent stepping under the tested contract.
* **State bounds:** positions, velocities, angles, angular velocities and energy remain finite. Energy remains in `[0, 100]` J. Angles remain in the implemented half-open interval `[-pi, pi)`.
* **Energy boundary behavior:** exhausted agents cannot actuate; agents that exhaust energy during a step complete that step and are subsequently inactive, with dead-agent state and energy behavior covered by tests.
* **One-step CPU/CUDA numerical agreement:** all compared continuous fields satisfy the strict test contract `rtol=1e-5, atol=1e-6`. Alive masks must match exactly.
* **40-second rollout:** run 2400 steps at the default `dt=1/60` on CPU and CUDA when available. Validate finite values, energy bounds and angle bounds at every step. Use identical seeded initial state, fixed actions and gravity on both devices. Report maximum absolute deviation per field and require identical alive masks. Long-rollout deviations are observational, not an additional cross-device tolerance gate.

## P0 Validation Evidence

### Test results

* Commit `c63fd68`: full suite passed on a free-Colab T4 runtime, 24/24 tests, including the one-step CPU/CUDA agreement test.
* Local CPU: 23 passed, 1 CUDA-dependent test skipped.
* CI: green at the validated commit.
* The T4-equipped full suite is not equivalent to every test executing on CUDA; most tests instantiate CPU states.
* The strict one-step CPU/CUDA comparison remains the numerical agreement contract. Long-rollout comparison is observational because small floating-point differences can accumulate.

### 40-second rollout evidence

* Initial rollout at `ee76bc2`: 2400 steps, 40.0 seconds, CPU and free T4.
* Both backends satisfied finite-state, energy `[0,100]` and angle `[-pi,pi)` bounds; alive masks matched.
* Observed maximum absolute deviations:

| Field | Maximum absolute deviation |
| --- | --- |
| Position | 0.0001220703125 m |
| Velocity | 3.814697265625e-06 m/s |
| Angle | 0.0 |
| Angular velocity | 0.0 |
| Energy | 0.0 |

* The stronger every-step validator is pending final execution on the exact freeze candidate.

### Benchmark evidence

Commit `c63fd68`, agent-steps/s, CPU vs. free-Colab T4:

| Agents | CPU | T4 |
| --- | --- | --- |
| 100 | 295,212 | 139,351 |
| 500 | 935,582 | 686,553 |
| 1,000 | 2,327,308 | 1,268,478 |
| 2,500 | 4,557,886 | 3,509,556 |
| 5,000 | 6,981,650 | 6,853,836 |

CPU is faster at the tested sizes below 5,000 agents; performance is approximately at parity at 5,000. These measurements do not indicate an optimization requirement.

Raw runtime, test, sanity and benchmark JSON: `benchmarks/results/p0_c63fd68_colab_t4.json`.

### Freeze procedure

Commit validation code and documentation without physics changes. Rerun the full suite, CPU validation, and the 40-second validator on the final candidate commit. Record the exact validated commit hash. Only then point `p0-validated` to that commit.

P0 validation freeze is not a research release or completion of LAWLESS. P1 proceeds under the original locked V1 design.

## Research integrity

Training, validation, and test environment seeds must be separated. Each result should record code revision, configuration, optimizer, controller, seed, and evaluation budget. A robust static policy must not be mistaken for within-lifetime adaptation; no-shift controls and plasticity-disabled ablations are required in later phases.

## Scope discipline

P0 intentionally excludes controllers, evolution, obstacles, lidar, objectives, and rendering. Those belong to later milestones after physics is validated.
