"""CUDA execution and CPU/CUDA parity gates for the P1.7 environment.

These tests are skipped only when CUDA is genuinely unavailable.  The
full-duration case exercises 2,400 actual environment transitions on CUDA.
"""

import pytest
import torch

from lawless.environment import reset, step
from lawless.physics.dynamics import rotate_gravity_90
from lawless.world.config import EnvironmentConfig, EpisodeConfig


_CUDA_SKIP = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is genuinely unavailable",
)

_FLOAT_RTOL = 1.0e-4
_FLOAT_ATOL = 2.0e-5


def _config(*, seed=42, duration=1.0, shift=0.5, dt=0.1, large=False):
    config = EnvironmentConfig(seed=seed)
    if large:
        config.world.width_m = 1_000_000.0
        config.world.height_m = 1_000_000.0
    else:
        config.world.width_m = 1000.0
        config.world.height_m = 1000.0
    config.world.obstacle_count = 0
    config.objectives.core_count = 0
    config.episode = EpisodeConfig(
        episode_duration_s=duration,
        shift_time_s=shift,
        dt_s=dt,
    )
    config.validate()
    return config


def _default_clock_config(*, seed=1, large=False):
    """Build the actual V1 default episode clock, independent of GPU access."""
    return _config(
        seed=seed,
        duration=40.0,
        shift=20.0,
        dt=1.0 / 60.0,
        large=large,
    )


def _assert_float_close(left, right):
    torch.testing.assert_close(
        left.detach().cpu(),
        right.detach().cpu(),
        rtol=_FLOAT_RTOL,
        atol=_FLOAT_ATOL,
    )


@_CUDA_SKIP
def test_cuda_environment_matches_cpu_discrete_events_masks_and_floating_outputs():
    config = _config(seed=117)
    cpu_state, cpu_obs, cpu_reset_info = reset(
        config, 2, 2, device="cpu", dtype=torch.float32
    )
    cuda_state, cuda_obs, cuda_reset_info = reset(
        config, 2, 2, device="cuda", dtype=torch.float32
    )

    assert cuda_obs.device.type == "cuda"
    assert all(
        world.episode.physics.position.device.type == "cuda"
        for world in cuda_state.worlds
    )
    for key in (
        "world_seeds",
        "condition_seeds",
        "shift_direction",
        "spawn_indices",
        "shift_applied_at_reset",
    ):
        assert torch.equal(cpu_reset_info[key], cuda_reset_info[key].cpu())
    for key in ("initial_gravity", "linear_damping_s_inv"):
        _assert_float_close(cpu_reset_info[key], cuda_reset_info[key])

    # Force one candidate in world zero to extract on the first transition;
    # all other candidates continue under the same independently sampled
    # world conditions. The edit is mirrored identically on CPU and CUDA.
    for state in (cpu_state, cuda_state):
        world = state.worlds[0]
        physics = world.episode.physics.clone()
        physics.position[0] = world.episode.generated_world.extraction_center
        episode = world.episode.replace_physics(physics)
        from dataclasses import replace
        from lawless.environment import EnvironmentState

        worlds = list(state.worlds)
        worlds[0] = replace(world, episode=episode)
        if state is cpu_state:
            cpu_state = EnvironmentState(tuple(worlds), config, 2)
        else:
            cuda_state = EnvironmentState(tuple(worlds), config, 2)

    actions_cpu = torch.tensor(
        [
            [[0.1, 0.0], [-0.2, 0.1]],
            [[0.0, -0.1], [0.3, 0.0]],
        ],
        dtype=torch.float32,
    )
    actions_cuda = actions_cpu.to("cuda")

    for _ in range(6):
        (
            cpu_state,
            cpu_obs,
            cpu_rewards,
            cpu_terminated,
            cpu_truncated,
            cpu_info,
        ) = step(cpu_state, actions_cpu)
        (
            cuda_state,
            cuda_obs,
            cuda_rewards,
            cuda_terminated,
            cuda_truncated,
            cuda_info,
        ) = step(cuda_state, actions_cuda)

        for key in (
            "newly_collected",
            "newly_extracted",
            "collision_contact",
            "unresolved_contact",
            "boundary_escape",
            "failure_this_step",
            "energy_death_this_step",
            "actuators_enabled",
            "shift_applied_this_step",
            "shift_applied",
            "world_completed",
            "truncated",
            "terminated",
        ):
            assert torch.equal(cpu_info[key], cuda_info[key].cpu()), key
        assert torch.equal(cpu_terminated, cuda_terminated.cpu())
        assert torch.equal(cpu_truncated, cuda_truncated.cpu())
        assert torch.equal(cpu_rewards, cuda_rewards.cpu())

        for key in (
            "energy_restored",
            "energy_spent",
            "gravity_used",
            "gravity_next",
            "elapsed_time_s",
        ):
            _assert_float_close(cpu_info[key], cuda_info[key])
        _assert_float_close(cpu_obs, cuda_obs)
        for world_index in range(2):
            cpu_physics = cpu_state.worlds[world_index].episode.physics
            cuda_physics = cuda_state.worlds[world_index].episode.physics
            for name in (
                "position",
                "velocity",
                "angle",
                "angular_velocity",
                "energy",
            ):
                _assert_float_close(
                    getattr(cpu_physics, name),
                    getattr(cuda_physics, name),
                )
            assert torch.equal(
                cpu_physics.alive,
                cuda_physics.alive.cpu(),
            )
            _assert_float_close(
                cpu_state.worlds[world_index].episode.embodiment.measured_acceleration,
                cuda_state.worlds[world_index].episode.embodiment.measured_acceleration,
            )


def test_cuda_default_duration_fixture_has_the_real_2400_transition_clock():
    # This preflight deliberately runs on CPU-only hosts too. It prevents the
    # CUDA gate from silently inheriting _config's short test defaults.
    config = _default_clock_config(seed=1, large=True)
    assert config.episode.dt_s == 1.0 / 60.0
    assert config.episode.shift_time_s == 20.0
    assert config.episode.episode_duration_s == 40.0
    assert round(config.episode.shift_time_s / config.episode.dt_s) == 1200
    assert round(
        config.episode.episode_duration_s / config.episode.dt_s
    ) == 2400


@_CUDA_SKIP
def test_cuda_executes_real_default_2400_transition_shift_and_deadline():
    config = _default_clock_config(seed=1, large=True)
    state, observations, reset_info = reset(
        config, 1, 1, device="cuda", dtype=torch.float32
    )
    assert config.episode.dt_s == 1.0 / 60.0
    assert config.episode.shift_time_s == 20.0
    assert config.episode.episode_duration_s == 40.0
    assert observations.device.type == "cuda"
    assert state.worlds[0].episode.physics.alive.all()

    initial_gravity = state.worlds[0].gravity.clone()
    shift_direction = state.worlds[0].shift_direction
    assert shift_direction in (-1, 1)
    shift_boundary = None
    for transition in range(1, 2401):
        state, observations, rewards, terminated, truncated, info = step(
            state, torch.zeros((1, 1, 2), device="cuda", dtype=torch.float32)
        )
        if transition == 1200:
            shift_boundary = info
            assert info["shift_applied_this_step"].item()
            assert torch.equal(info["gravity_used"][0], initial_gravity)
            expected_rotated = rotate_gravity_90(initial_gravity, shift_direction)
            torch.testing.assert_close(
                info["gravity_next"][0], expected_rotated, rtol=0, atol=0
            )
            torch.testing.assert_close(
                torch.linalg.vector_norm(info["gravity_next"][0]),
                torch.linalg.vector_norm(initial_gravity),
                rtol=0,
                atol=0,
            )
            assert state.worlds[0].step_count == 1200
            assert state.worlds[0].episode.physics.alive.all()
        elif transition == 1201:
            assert not info["shift_applied_this_step"].item()
            assert torch.equal(
                info["gravity_used"][0], shift_boundary["gravity_next"][0]
            )
        if transition < 2400:
            assert not terminated.any()
            assert not truncated.any()
            assert state.worlds[0].episode.physics.alive.all()
            assert not state.worlds[0].completed

    assert state.worlds[0].step_count == 2400
    assert state.worlds[0].completed
    assert info["elapsed_steps"].item() == 2400
    assert info["elapsed_time_s"].item() == 40.0
    assert not terminated.item()
    assert truncated.item()
    assert state.worlds[0].episode.physics.alive.item()
    assert info["shift_applied"].item()
    assert state.worlds[0].gravity.device.type == "cuda"

    with pytest.raises(RuntimeError, match="complete"):
        step(state, torch.zeros((1, 1, 2), device="cuda", dtype=torch.float32))
    assert state.worlds[0].step_count == 2400
