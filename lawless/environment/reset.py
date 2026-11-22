"""Deterministic construction of B independent LAWLESS episode worlds."""

from __future__ import annotations

from dataclasses import replace
import math
from numbers import Integral

import torch
from torch import Tensor

from lawless.physics.constants import PhysicsConfig
from lawless.physics.dynamics import rotate_gravity_90
from lawless.sensors.lidar import sense_lidar
from lawless.sensors.observations import assemble_observations
from lawless.world.config import EnvironmentConfig
from lawless.world.episode_state import AgentEpisodeState
from lawless.world.full_generator import generate_world
from lawless.environment.episode import EnvironmentState, WorldEpisodeState


_MASK64 = (1 << 64) - 1
_MAX_SEED = (1 << 63) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_WORLD_STREAM = 0x574F524C445F5631
_CONDITION_STREAM = 0x434F4E445F5631


def splitmix64(value: int) -> int:
    """Return the SplitMix64 finalizer of an unsigned 64-bit input."""
    z = (int(value) + _GOLDEN) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def derive_stream_seed(root_seed: int, world_index: int, stream: int) -> int:
    """Derive a stable nonnegative 63-bit seed for one world/stream pair.

    The root seed, zero-based world index, and fixed stream identifier are
    mixed without depending on B, N, iteration order, or another RNG stream.
    """
    if isinstance(root_seed, bool) or not isinstance(root_seed, Integral):
        raise TypeError("root_seed must be an integer")
    if not 0 <= int(root_seed) <= _MAX_SEED:
        raise ValueError("root_seed must be in [0, 2**63 - 1]")
    if isinstance(world_index, bool) or not isinstance(world_index, Integral):
        raise TypeError("world_index must be an integer")
    if int(world_index) < 0:
        raise ValueError("world_index must be nonnegative")
    if isinstance(stream, bool) or not isinstance(stream, Integral):
        raise TypeError("stream must be an integer")
    mixed = (
        int(root_seed)
        + _GOLDEN * (int(world_index) + 1)
        + int(stream)
    ) & _MASK64
    return splitmix64(mixed) & _MAX_SEED


def derive_world_seed(root_seed: int, world_index: int) -> int:
    return derive_stream_seed(root_seed, world_index, _WORLD_STREAM)


def derive_condition_seed(root_seed: int, world_index: int) -> int:
    return derive_stream_seed(root_seed, world_index, _CONDITION_STREAM)


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _sample_conditions(
    config: EnvironmentConfig,
    *,
    condition_seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, float, int]:
    """Sample initial gravity, damping, and shift sign from a private CPU RNG."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(condition_seed)
    direction_index = int(torch.randint(
        len(config.episode.gravity_directions), (1,), generator=generator
    ).item())
    damping_index = int(torch.randint(
        len(config.episode.linear_damping_s_inv), (1,), generator=generator
    ).item())
    shift_sign = 1 if int(torch.randint(2, (1,), generator=generator).item()) else -1

    direction = config.episode.gravity_directions[direction_index]
    gravity = torch.tensor(
        direction,
        dtype=dtype,
        device=device,
    ) * float(config.episode.gravity_m_s2)
    damping = float(config.episode.linear_damping_s_inv[damping_index])
    return gravity, damping, shift_sign


def _observe_world(world_state: WorldEpisodeState, sensor_config) -> Tensor:
    episode = world_state.episode
    lidar = sense_lidar(
        episode.physics,
        episode.generated_world.world,
        sensor_config,
    )
    return assemble_observations(episode, lidar)


def reset(
    config: EnvironmentConfig,
    num_worlds: int,
    num_agents: int,
    *,
    physics_config: PhysicsConfig | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[EnvironmentState, Tensor, dict[str, Tensor]]:
    """Create B independent worlds, each containing N paired candidate rollouts.

    Every world generates exactly one spawn slot. All N candidates in that
    world receive explicit spawn index zero and independent physical,
    lifecycle, measurement, and objective tensors. The static layout is
    shared by reference within that world only.

    World geometry and episode-condition sampling use separate private CPU
    generators with independently derived seeds. Neither changes global RNG.
    """
    if not isinstance(config, EnvironmentConfig):
        raise TypeError("config must be an EnvironmentConfig")
    config.validate()
    b = _positive_integer("num_worlds", num_worlds)
    n = _positive_integer("num_agents", num_agents)
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("dtype must be torch.float32 or torch.float64")
    try:
        target_device = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"invalid device: {device!r}") from exc
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if target_device.type not in ("cpu", "cuda"):
        raise ValueError("device must be CPU or CUDA")
    if physics_config is None:
        physics_config = PhysicsConfig()
    if not isinstance(physics_config, PhysicsConfig):
        raise TypeError("physics_config must be a PhysicsConfig")

    episode_steps = config.episode.episode_duration_s / config.episode.dt_s
    shift_steps = config.episode.shift_time_s / config.episode.dt_s
    if not math.isclose(episode_steps, round(episode_steps), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("episode duration must align with an integer timestep")
    if not math.isclose(shift_steps, round(shift_steps), rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("gravity shift must align with an integer timestep")
    if not math.isclose(physics_config.dt, config.episode.dt_s, rel_tol=0.0, abs_tol=1e-12):
        # EpisodeConfig is the environment clock. P0 receives that exact dt.
        physics_config = replace(physics_config, dt=float(config.episode.dt_s))

    worlds: list[WorldEpisodeState] = []
    world_seeds: list[int] = []
    condition_seeds: list[int] = []
    initial_gravities: list[Tensor] = []
    damping_values: list[float] = []
    shift_directions: list[int] = []

    for world_index in range(b):
        world_seed = derive_world_seed(config.seed, world_index)
        condition_seed = derive_condition_seed(config.seed, world_index)
        generated = generate_world(
            config,
            seed=world_seed,
            spawn_count=1,
            device=target_device,
            dtype=dtype,
        )
        gravity, damping, shift_direction = _sample_conditions(
            config,
            condition_seed=condition_seed,
            device=target_device,
            dtype=dtype,
        )
        initial_gravity = gravity.clone()
        shift_at_reset = config.episode.shift_time_s == 0.0
        if shift_at_reset:
            gravity = rotate_gravity_90(gravity, shift_direction)
        per_world_physics = replace(
            physics_config,
            dt=float(config.episode.dt_s),
            linear_damping=damping,
            angular_damping=float(config.episode.angular_damping_s_inv),
        )
        spawn_indices = torch.zeros(
            (n,), dtype=torch.int64, device=target_device
        )
        episode = AgentEpisodeState.initialize(
            generated,
            spawn_indices,
            embodiment_config=config.embodiment,
            objective_config=config.objectives,
            physics_config=per_world_physics,
        )
        world_state = WorldEpisodeState(
            episode=episode,
            physics_config=per_world_physics,
            gravity=gravity,
            initial_gravity=initial_gravity,
            shift_direction=shift_direction,
            step_count=0,
            shift_applied=shift_at_reset,
            truncated=torch.zeros((n, 1), dtype=torch.bool, device=target_device),
            completed=False,
            world_seed=world_seed,
            condition_seed=condition_seed,
            linear_damping_s_inv=damping,
        )
        worlds.append(world_state)
        world_seeds.append(world_seed)
        condition_seeds.append(condition_seed)
        initial_gravities.append(initial_gravity)
        damping_values.append(damping)
        shift_directions.append(shift_direction)

    environment = EnvironmentState(
        worlds=tuple(worlds),
        config=config,
        num_agents=n,
    )
    observations = torch.stack(
        [_observe_world(world, config.sensors) for world in worlds],
        dim=0,
    )
    info = {
        "world_seeds": torch.tensor(world_seeds, dtype=torch.int64, device=target_device),
        "condition_seeds": torch.tensor(condition_seeds, dtype=torch.int64, device=target_device),
        "initial_gravity": torch.stack(initial_gravities, dim=0),
        "linear_damping_s_inv": torch.tensor(damping_values, dtype=dtype, device=target_device),
        "shift_direction": torch.tensor(shift_directions, dtype=torch.int64, device=target_device),
        "shift_applied_at_reset": torch.full(
            (b,), config.episode.shift_time_s == 0.0,
            dtype=torch.bool, device=target_device,
        ),
        "spawn_indices": torch.zeros((b, n), dtype=torch.int64, device=target_device),
    }
    return environment, observations, info
