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

## P0 acceptance criteria

- Analytical first-step dynamics match expected values.
- No NaN/Inf state values in a 40-second simulation.
- Energy remains in [0, initial_energy].
- Angles remain in [-pi, pi).
- Repeated CPU runs with identical initial state and actions are identical.
- Batched stepping matches independent single-agent stepping within tolerance.
- CPU/GPU benchmark records throughput and max numerical deviation; performance thresholds are reported, not fabricated.

## Research integrity

Training, validation, and test environment seeds must be separated. Each result should record code revision, configuration, optimizer, controller, seed, and evaluation budget. A robust static policy must not be mistaken for within-lifetime adaptation; no-shift controls and plasticity-disabled ablations are required in later phases.

## Scope discipline

P0 intentionally excludes controllers, evolution, obstacles, lidar, objectives, and rendering. Those belong to later milestones after physics is validated.
