# LAWLESS P1.0 — Environment Configuration Foundation

## Status

P1.0 implements the validated environment configuration
foundation only.

It introduces:

- Validated environment configuration dataclasses.
- YAML configuration loading.
- Strict field and section validation.
- Provisional environment defaults.
- Configuration validation tests.

It does not implement:

- Generated world geometry.
- Obstacle placement.
- Collision detection or response.
- Energy-core placement or interaction.
- Lidar simulation.
- Observation generation.
- Episode execution.
- Controllers.
- Evolution or optimization.

P1.0 is a configuration-only stage.

---

## Configuration structure

The configuration is organized into:

- `WorldConfig`
- `EmbodimentConfig`
- `ObjectiveConfig`
- `SensorConfig`
- `EpisodeConfig`

These are composed into `EnvironmentConfig`.

---

## Units

| Parameter | Unit |
|---|---|
| World dimensions | m |
| Obstacle radii | m |
| Agent radius | m |
| Core radius | m |
| Core energy | J |
| Extraction radius | m |
| Lidar range | m |
| Episode duration | s |
| Shift time | s |
| Integration timestep | s |
| Gravity magnitude | m/s² |
| Linear damping | s⁻¹ |
| Angular damping | s⁻¹ |

---

## Provisional defaults

### World

- Width: 100 m
- Height: 100 m
- Obstacle count: 20
- Obstacle radius: 1–3 m

### Embodiment

- Agent radius: 0.5 m

### Objectives

- Core count: 8
- Core radius: 0.75 m
- Core energy: 25 J
- Extraction radius: 3 m

### Sensors

- Lidar ray count: exactly 8
- Lidar range: 20 m

The eight-ray count is fixed to preserve the locked P1
sensor/observation contract.

### Episode

- Duration: 40 s
- Shift time: 20 s
- Timestep: 1/60 s
- Gravity magnitude: 4 m/s²
- Angular damping: 0.20 s⁻¹

Candidate gravity directions:

- (0, 1)
- (1, 0)
- (0, -1)
- (-1, 0)

Candidate linear damping:

- 0.05 s⁻¹
- 0.15 s⁻¹
- 0.30 s⁻¹

These are provisional P1 environment parameters. They do
not change the frozen P0 physics contract.

---

## Validation

The configuration validator enforces:

- Finite numeric values.
- Positive world dimensions.
- Positive radii and physical configuration magnitudes.
- Nonnegative obstacle and core counts.
- Strict integer counts, excluding boolean values.
- Integer seed, excluding boolean values.
- Exactly eight lidar rays.
- Valid gravity unit vectors.
- Nonnegative damping.
- Valid episode timing.
- Integration-step alignment.
- Correct configuration section types.
- Correct list/tuple container types.
- Rejection of unknown configuration keys.

Zero obstacle count and zero core count are valid.

---

## Installation

From the repository root, PyYAML is required.

LAWLESS already declares this dependency in its existing
project dependency files.

If your environment is not installed yet, use the existing
repository dependency installation workflow.

---

## Configuration loading

```python
from lawless.world.config import load_config

config = load_config("configs/environment.yaml")

print(config.world.width_m)
print(config.episode.dt_s)
print(config.episode.gravity_directions)
print(config.sensors.lidar_ray_count)
```

The loader validates the configuration immediately.

Unknown keys and malformed configuration sections are
rejected instead of being silently accepted.

---

## Tests

Run the P1.0 validation tests from the repository root:

```bash
pytest -q tests/test_p1_config.py
```

Run the entire repository suite:

```bash
pytest -q
```

Local validation: The P1.0 configuration tests pass (148 passed in 0.13s), and the full repository suite passes (171 passed, 1 expected CUDA skip, in 0.51s); this stage adds configuration and validation only, with no P1.0 T4 or CI result claimed.

---

## Files

```text
lawless/
    world/
        __init__.py
        config.py
    environment/
        __init__.py

configs/
    environment.yaml

tests/
    test_p1_config.py

README_P1_0.md
```

---

## P1.0 completion gate

P1.0 is complete only after:

1. The replacement configuration file is integrated.
2. The expanded tests are integrated.
3. All P1.0 tests pass.
4. The full repository test suite passes.
5. P0 remains unchanged.

This stage does not claim a functional environment or
research readiness.

---

## Next stage

P1.1 — continuous world geometry and boundaries.

P1.1 begins only after the P1.0 validation gate is
confirmed.
