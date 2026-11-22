"""P1.7 environment acceptance coverage over the composed reset/step API.

These tests exercise the real default episode clock, composed collision and
objective paths, deterministic independent-world execution, and observation
privacy.  They intentionally do not modify or replace accepted P0/P1 tests.
"""

from dataclasses import replace
import math

import pytest
import torch

from lawless.environment import EnvironmentState, reset, step
from lawless.physics.dynamics import rotate_gravity_90
from lawless.sensors.normalization import (
    ANGULAR_VELOCITY_SCALE_RADPS,
    LINEAR_ACCELERATION_SCALE_MPS2,
    LINEAR_VELOCITY_SCALE_MPS,
)
from lawless.world.config import EnvironmentConfig, EpisodeConfig
from lawless.world.geometry import CircleObstacles
from lawless.world.world_state import WorldState


_FLOAT_RTOL = 1.0e-5
_FLOAT_ATOL = 1.0e-6


def _config(
    *,
    seed=42,
    duration=1.0,
    shift=0.5,
    dt=0.1,
    cores=0,
    width=100.0,
    height=100.0,
    obstacles=0,
):
    config = EnvironmentConfig(seed=seed)
    config.world.width_m = width
    config.world.height_m = height
    config.world.obstacle_count = obstacles
    config.objectives.core_count = cores
    config.episode = EpisodeConfig(
        episode_duration_s=duration,
        shift_time_s=shift,
        dt_s=dt,
    )
    config.validate()
    return config


def _replace_world(environment, index, world):
    worlds = list(environment.worlds)
    worlds[index] = world
    return EnvironmentState(tuple(worlds), environment.config, environment.num_agents)


def _with_zero_gravity(world):
    return replace(world, gravity=torch.zeros_like(world.gravity))


def _replace_physics(world, physics):
    return replace(world, episode=world.episode.replace_physics(physics))


def _safe_obstacle_center(generated_world, width, height):
    """Choose a deterministic fixture point clear of the generated spawn/target."""
    spawn = generated_world.spawn_positions[0]
    extraction = generated_world.extraction_center
    candidates = [
        (x, y)
        for x in (100.0, 300.0, 500.0, 700.0, 900.0)
        for y in (100.0, 300.0, 500.0, 700.0, 900.0)
        if x < width and y < height
    ]
    for x, y in candidates:
        center = torch.tensor([x, y], dtype=spawn.dtype, device=spawn.device)
        if (
            torch.linalg.vector_norm(center - spawn) > 20.0
            and torch.linalg.vector_norm(center - extraction) > 20.0
        ):
            return center
    raise AssertionError("fixture could not find a clear obstacle location")


def _assert_observation_acceleration(observation, world):
    physics = world.episode.physics
    acceleration = world.episode.embodiment.measured_acceleration
    angle = physics.angle
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    expected_x = (
        acceleration[..., 0:1] * cosine
        + acceleration[..., 1:2] * sine
    ) / LINEAR_ACCELERATION_SCALE_MPS2
    expected_y = (
        -acceleration[..., 0:1] * sine
        + acceleration[..., 1:2] * cosine
    ) / LINEAR_ACCELERATION_SCALE_MPS2
    expected = torch.cat((expected_x, expected_y), dim=-1).clamp(-1.0, 1.0)
    torch.testing.assert_close(
        observation[..., 20:22],
        expected,
        rtol=_FLOAT_RTOL,
        atol=_FLOAT_ATOL,
    )


def test_default_2400_transition_episode_shift_signs_and_deadline():
    """Run the actual default 40 s / 1/60 s episode, not a shortened clock."""
    config = EnvironmentConfig(seed=1)
    config.world.width_m = 1_000_000.0
    config.world.height_m = 1_000_000.0
    config.world.obstacle_count = 0
    config.objectives.core_count = 0
    config.validate()

    environment, observations, reset_info = reset(
        config, num_worlds=2, num_agents=2, dtype=torch.float64
    )
    assert config.episode.dt_s == 1.0 / 60.0
    assert config.episode.shift_time_s == 20.0
    assert config.episode.episode_duration_s == 40.0
    assert observations.shape == (2, 2, 23)
    assert [world.step_count for world in environment.worlds] == [0, 0]
    assert [world.shift_direction for world in environment.worlds] == [1, -1]
    assert reset_info["shift_direction"].tolist() == [1, -1]
    replay_state, _, replay_info = reset(
        config, num_worlds=2, num_agents=2, dtype=torch.float64
    )
    assert torch.equal(reset_info["world_seeds"], replay_info["world_seeds"])
    assert torch.equal(reset_info["condition_seeds"], replay_info["condition_seeds"])
    assert torch.equal(reset_info["shift_direction"], replay_info["shift_direction"])
    for first_world, replay_world in zip(environment.worlds, replay_state.worlds):
        torch.testing.assert_close(
            first_world.gravity, replay_world.gravity, rtol=0, atol=0
        )

    initial_gravity = [world.gravity.clone() for world in environment.worlds]
    initial_damping = [
        world.physics_config.linear_damping for world in environment.worlds
    ]
    initial_magnitudes = [torch.linalg.vector_norm(value) for value in initial_gravity]

    boundary_info = None
    for transition in range(1, 1201):
        environment, observations, _, terminated, truncated, info = step(
            environment,
            torch.zeros((2, 2, 2), dtype=torch.float64),
        )
        assert not terminated.any()
        assert not truncated.any()
        assert all(world.episode.physics.alive.all() for world in environment.worlds)
        assert not any(world.completed for world in environment.worlds)
        if transition == 1200:
            boundary_info = info

    assert boundary_info is not None
    for world_index, world in enumerate(environment.worlds):
        expected_rotated = rotate_gravity_90(
            initial_gravity[world_index], world.shift_direction
        )
        torch.testing.assert_close(
            boundary_info["gravity_used"][world_index],
            initial_gravity[world_index],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            boundary_info["gravity_next"][world_index],
            expected_rotated,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            world.gravity, expected_rotated, rtol=0, atol=0
        )
        assert boundary_info["shift_applied_this_step"][world_index].item()
        assert boundary_info["shift_applied"][world_index].item()
        assert world.shift_applied
        assert world.step_count == 1200
        assert world.physics_config.linear_damping == initial_damping[world_index]
        assert world.linear_damping_s_inv == initial_damping[world_index]
        torch.testing.assert_close(
            torch.linalg.vector_norm(world.gravity),
            initial_magnitudes[world_index],
            rtol=0,
            atol=1e-12,
        )

    # Transition 1201 uses the already-rotated gravity.
    environment, observations, _, terminated, truncated, info1201 = step(
        environment, torch.zeros((2, 2, 2), dtype=torch.float64)
    )
    for world_index, world in enumerate(environment.worlds):
        torch.testing.assert_close(
            info1201["gravity_used"][world_index],
            boundary_info["gravity_next"][world_index],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            info1201["gravity_next"][world_index],
            boundary_info["gravity_next"][world_index],
            rtol=0,
            atol=0,
        )
        assert not info1201["shift_applied_this_step"][world_index].item()
        assert world.step_count == 1201

    # Leave one candidate inside the extraction radius only for transition
    # 2400. Its extraction must win over timeout; its paired candidate remains
    # physically alive and is truncated at the deadline.
    for _ in range(1201, 2399):
        environment, observations, _, terminated, truncated, _ = step(
            environment, torch.zeros((2, 2, 2), dtype=torch.float64)
        )
        assert not terminated.any()
        assert not truncated.any()

    world0 = environment.worlds[0]
    physics = world0.episode.physics.clone()
    physics.position[0] = world0.episode.generated_world.extraction_center
    physics.velocity[0].zero_()
    environment = _replace_world(
        environment, 0, _replace_physics(world0, physics)
    )

    environment, observations, rewards, terminated, truncated, deadline_info = step(
        environment, torch.zeros((2, 2, 2), dtype=torch.float64)
    )

    assert [world.step_count for world in environment.worlds] == [2400, 2400]
    assert [world.completed for world in environment.worlds] == [True, True]
    assert deadline_info["elapsed_steps"].tolist() == [2400, 2400]
    assert deadline_info["elapsed_time_s"].tolist() == [40.0, 40.0]
    assert deadline_info["newly_extracted"].tolist() == [[True, False], [False, False]]
    assert rewards.tolist() == [[1.0, 0.0], [0.0, 0.0]]
    assert terminated.tolist() == [[True, False], [False, False]]
    assert truncated.tolist() == [[False, True], [True, True]]
    assert deadline_info["shift_applied"].all()
    assert all(
        world.physics_config.linear_damping == initial_damping[index]
        for index, world in enumerate(environment.worlds)
    )

    # A 2401st transition is rejected, and the input state remains unchanged.
    final_positions = [
        world.episode.physics.position.clone() for world in environment.worlds
    ]
    with pytest.raises(RuntimeError, match="complete"):
        step(environment, torch.zeros((2, 2, 2), dtype=torch.float64))
    for world, before in zip(environment.worlds, final_positions):
        assert world.step_count == 2400
        torch.testing.assert_close(world.episode.physics.position, before, rtol=0, atol=0)


def test_environment_collision_resolution_drives_measured_acceleration_and_heading_observation():
    config = _config(
        duration=1.0, shift=0.5, dt=0.1, width=1000.0, height=1000.0
    )
    environment, _, _ = reset(config, 1, 1, dtype=torch.float64)
    world = _with_zero_gravity(environment.worlds[0])

    # A real static obstacle is installed in the generated world's geometry.
    # The agent's P0 endpoint penetrates it, so the resolver removes inward
    # velocity before the environment records finite-difference acceleration.
    center = _safe_obstacle_center(
        world.episode.generated_world, config.world.width_m, config.world.height_m
    )
    obstacle_world = WorldState(
        config.world.width_m,
        config.world.height_m,
        CircleObstacles(
            centers=center.reshape(1, 2),
            radii=torch.tensor([1.0], dtype=torch.float64),
        ),
    )
    generated = replace(world.episode.generated_world, world=obstacle_world)
    episode = replace(world.episode, generated_world=generated)
    physics = episode.physics.clone()
    physics.position[0] = center + torch.tensor([1.51, 0.0], dtype=torch.float64)
    physics.velocity[0] = torch.tensor([-1.0, 0.0], dtype=torch.float64)
    physics.angle[0, 0] = math.pi / 2.0
    episode = episode.replace_physics(physics)
    world = replace(world, episode=episode)
    environment = _replace_world(environment, 0, world)

    environment, observations, _, terminated, truncated, info = step(
        environment, torch.zeros((1, 1, 2), dtype=torch.float64)
    )
    next_world = environment.worlds[0]
    measured = next_world.episode.embodiment.measured_acceleration

    assert info["collision_contact"].item()
    assert not info["unresolved_contact"].item()
    assert not info["boundary_escape"].item()
    assert not info["failure_this_step"].item()
    assert not terminated.item() and not truncated.item()
    torch.testing.assert_close(
        next_world.episode.physics.velocity,
        torch.zeros((1, 2), dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        measured,
        torch.tensor([[10.0, 0.0]], dtype=torch.float64),
        rtol=0,
        atol=1e-10,
    )
    _assert_observation_acceleration(observations[0], next_world)
    torch.testing.assert_close(
        observations[0, 0, 20:22],
        torch.tensor([0.0, -10.0 / 16.0], dtype=torch.float64),
        rtol=1e-8,
        atol=1e-8,
    )


def test_environment_boundary_escape_is_reported_and_annotated_as_failure():
    config = _config()
    environment, _, _ = reset(config, 1, 1, dtype=torch.float64)
    world = _with_zero_gravity(environment.worlds[0])
    physics = world.episode.physics.clone()
    physics.position[0] = torch.tensor([-0.1, 50.0], dtype=torch.float64)
    physics.velocity.zero_()
    world = _replace_physics(world, physics)
    environment = _replace_world(environment, 0, world)

    environment, _, rewards, terminated, truncated, info = step(
        environment, torch.zeros((1, 1, 2), dtype=torch.float64)
    )
    assert info["boundary_escape"].item()
    assert info["failure_this_step"].item()
    assert not info["unresolved_contact"].item()
    assert environment.worlds[0].episode.embodiment.failed.item()
    assert not environment.worlds[0].episode.embodiment.extracted.item()
    assert terminated.item()
    assert not truncated.item()
    assert rewards.item() == 0.0
    assert environment.worlds[0].completed


def test_environment_unresolved_obstacle_contact_is_reported_and_marks_failure():
    config = _config(width=1000.0, height=1000.0)
    environment, _, _ = reset(config, 1, 1, dtype=torch.float64)
    world = _with_zero_gravity(environment.worlds[0])

    # The obstacle circles themselves are valid and strictly separated.
    # Their expanded agent-clearance regions overlap, leaving the agent
    # unresolved after the resolver's bounded ordered passes.
    center = _safe_obstacle_center(
        world.episode.generated_world, config.world.width_m, config.world.height_m
    )
    obstacles = CircleObstacles(
        centers=torch.stack(
            (
                center + torch.tensor([-1.005, 0.0], dtype=torch.float64),
                center + torch.tensor([1.005, 0.0], dtype=torch.float64),
            )
        ),
        radii=torch.tensor([1.0, 1.0], dtype=torch.float64),
    )
    collision_world = WorldState(
        config.world.width_m, config.world.height_m, obstacles
    )
    generated = replace(world.episode.generated_world, world=collision_world)
    episode = replace(world.episode, generated_world=generated)
    physics = episode.physics.clone()
    physics.position[0] = center
    physics.velocity.zero_()
    episode = episode.replace_physics(physics)
    world = replace(world, episode=episode)
    environment = _replace_world(environment, 0, world)

    environment, _, rewards, terminated, truncated, info = step(
        environment, torch.zeros((1, 1, 2), dtype=torch.float64)
    )
    assert info["unresolved_contact"].item()
    assert info["collision_contact"].item()
    assert not info["boundary_escape"].item()
    assert info["failure_this_step"].item()
    assert not info["newly_extracted"].item()
    assert rewards.item() == 0.0
    assert environment.worlds[0].episode.embodiment.failed.item()
    assert not environment.worlds[0].episode.embodiment.extracted.item()
    assert terminated.item() and not truncated.item()


def test_p0_energy_death_takes_precedence_over_collision_failure_annotation():
    config = _config()
    environment, _, _ = reset(config, 1, 1, dtype=torch.float64)
    world = _with_zero_gravity(environment.worlds[0])
    physics = world.episode.physics.clone()
    # Keep the P0-thrusted endpoint outside the world so the collision
    # report coincides with energy exhaustion on this same transition.
    physics.position[0] = torch.tensor([-2.0, 50.0], dtype=torch.float64)
    physics.velocity.zero_()
    physics.energy[0, 0] = 0.001
    world = _replace_physics(world, physics)
    environment = _replace_world(environment, 0, world)

    environment, _, _, terminated, truncated, info = step(
        environment,
        torch.tensor([[[1.0, 0.0]]], dtype=torch.float64),
    )
    assert info["energy_death_this_step"].item()
    assert info["boundary_escape"].item()
    assert not info["failure_this_step"].item()
    assert not environment.worlds[0].episode.embodiment.failed.item()
    assert not environment.worlds[0].episode.embodiment.extracted.item()
    assert not environment.worlds[0].episode.physics.alive.item()
    assert terminated.item() and not truncated.item()


def test_environment_core_collection_is_once_only_capped_and_candidate_owned():
    config = _config(cores=1)
    environment, _, _ = reset(config, 1, 2, dtype=torch.float64)
    world = _with_zero_gravity(environment.worlds[0])
    core = world.episode.generated_world.core_positions[0].clone()
    physics = world.episode.physics.clone()
    physics.position[0] = core
    physics.energy[:] = 99.75
    world = _replace_physics(world, physics)
    environment = _replace_world(environment, 0, world)

    environment, _, rewards1, _, _, info1 = step(
        environment, torch.zeros((1, 2, 2), dtype=torch.float64)
    )
    assert info1["newly_collected"].tolist() == [[[True], [False]]]
    assert info1["core_collection_count"].tolist() == [[1, 0]]
    torch.testing.assert_close(
        info1["energy_restored"],
        torch.tensor([[0.25, 0.0]], dtype=torch.float64),
        rtol=0,
        atol=1e-10,
    )
    assert rewards1.tolist() == [[0.0, 0.0]]
    assert environment.worlds[0].episode.objectives.collected[:, 0].tolist() == [
        True, False
    ]
    assert environment.worlds[0].episode.physics.energy[0, 0].item() == 100.0

    # Candidate one now contacts the same core. Candidate zero cannot collect
    # it again, while candidate one's independent mask permits its own event.
    world = environment.worlds[0]
    physics = world.episode.physics.clone()
    physics.position[1] = core
    world = _replace_physics(world, physics)
    environment = _replace_world(environment, 0, world)
    environment, _, _, _, _, info2 = step(
        environment, torch.zeros((1, 2, 2), dtype=torch.float64)
    )
    assert info2["newly_collected"].tolist() == [[[False], [True]]]
    assert info2["core_collection_count"].tolist() == [[0, 1]]
    assert info2["energy_restored"][0, 0].item() == 0.0
    assert info2["energy_restored"][0, 1].item() == pytest.approx(0.25)
    assert environment.worlds[0].episode.objectives.collected[:, 0].tolist() == [
        True, True
    ]


def test_batched_worlds_match_separately_stepped_world_state_references():
    config = _config(seed=710, duration=2.0, shift=1.0, dt=0.1)
    batched, _, reset_info = reset(config, 3, 2, dtype=torch.float64)
    references = [
        EnvironmentState((world,), config, num_agents=2)
        for world in batched.worlds
    ]
    for index, world in enumerate(batched.worlds):
        assert world.world_seed == reset_info["world_seeds"][index].item()
        assert world.condition_seed == reset_info["condition_seeds"][index].item()
        torch.testing.assert_close(
            world.initial_gravity,
            reset_info["initial_gravity"][index],
            rtol=0,
            atol=0,
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(98765)
    for _ in range(7):
        actions = torch.rand((3, 2, 2), generator=generator, dtype=torch.float64) * 2 - 1
        batched, batch_obs, batch_rewards, batch_term, batch_trunc, batch_info = step(
            batched, actions
        )
        for index in range(3):
            (
                references[index],
                single_obs,
                single_rewards,
                single_term,
                single_trunc,
                single_info,
            ) = step(references[index], actions[index:index + 1])
            torch.testing.assert_close(
                batch_obs[index:index + 1], single_obs, rtol=0, atol=0
            )
            torch.testing.assert_close(
                batch_rewards[index:index + 1], single_rewards, rtol=0, atol=0
            )
            assert torch.equal(batch_term[index:index + 1], single_term)
            assert torch.equal(batch_trunc[index:index + 1], single_trunc)
            for key, value in single_info.items():
                torch.testing.assert_close(
                    batch_info[key][index:index + 1],
                    value,
                    rtol=0,
                    atol=0,
                )
            torch.testing.assert_close(
                batched.worlds[index].episode.physics.position,
                references[index].worlds[0].episode.physics.position,
                rtol=0,
                atol=0,
            )


def test_step_does_not_consume_global_rng_and_action_sequence_replays():
    config = _config(seed=991, duration=2.0, shift=1.0, dt=0.1)
    first, _, _ = reset(config, 2, 3, dtype=torch.float64)
    second, _, _ = reset(config, 2, 3, dtype=torch.float64)

    action_generator = torch.Generator(device="cpu")
    action_generator.manual_seed(5512)
    action_sequence = [
        torch.rand((2, 3, 2), generator=action_generator, dtype=torch.float64) * 2 - 1
        for _ in range(5)
    ]

    torch.manual_seed(123456)
    expected_global = torch.rand(8)
    torch.manual_seed(123456)
    global_state_before = torch.random.get_rng_state().clone()

    for action_index, actions in enumerate(action_sequence):
        first, obs_a, reward_a, term_a, trunc_a, info_a = step(first, actions)
        if action_index == 0:
            global_state_after = torch.random.get_rng_state()
            assert torch.equal(global_state_before, global_state_after)
        second, obs_b, reward_b, term_b, trunc_b, info_b = step(second, actions)
        torch.testing.assert_close(obs_a, obs_b, rtol=0, atol=0)
        torch.testing.assert_close(reward_a, reward_b, rtol=0, atol=0)
        assert torch.equal(term_a, term_b)
        assert torch.equal(trunc_a, trunc_b)
        for key in info_a:
            torch.testing.assert_close(info_a[key], info_b[key], rtol=0, atol=0)

    assert torch.equal(torch.rand(8), expected_global)


def test_hidden_episode_metadata_is_not_an_observation_channel():
    config = _config(seed=19)
    environment, observations, _ = reset(config, 1, 2, dtype=torch.float64)
    world = environment.worlds[0]
    alternate_damping = 0.30 if world.linear_damping_s_inv != 0.30 else 0.05
    alternate_physics_config = replace(
        world.physics_config, linear_damping=alternate_damping
    )
    alternate = replace(
        world,
        gravity=-world.gravity,
        initial_gravity=-world.initial_gravity,
        shift_direction=-world.shift_direction,
        shift_applied=True,
        physics_config=alternate_physics_config,
        linear_damping_s_inv=alternate_damping,
    )
    alternate_environment = EnvironmentState(
        (alternate,), config, num_agents=2
    )
    from lawless.environment.reset import _observe_world

    alternate_observations = _observe_world(alternate, config.sensors)
    torch.testing.assert_close(
        observations[0], alternate_observations, rtol=0, atol=0
    )
    assert observations.shape == (1, 2, 23)
    assert alternate_observations.shape == (2, 23)
    # The alternate world is a valid state with different hidden conditions,
    # but the same physical/objective snapshot; no condition metadata is
    # exposed as an extra observation channel.
    assert alternate_environment.worlds[0].step_count == 0


def test_fixed_observation_normalization_remains_valid_across_gravity_shift():
    config = _config(seed=23, duration=1.0, shift=0.1, dt=0.1)
    environment, observations, _ = reset(config, 1, 1, dtype=torch.float64)
    world = environment.worlds[0]
    initial_gravity = world.gravity.clone()

    environment, observations1, _, _, _, info1 = step(
        environment, torch.zeros((1, 1, 2), dtype=torch.float64)
    )
    world1 = environment.worlds[0]
    assert info1["shift_applied_this_step"].item()
    torch.testing.assert_close(info1["gravity_used"][0], initial_gravity, rtol=0, atol=0)
    _assert_observation_acceleration(observations1[0], world1)

    environment, observations2, _, _, _, info2 = step(
        environment, torch.zeros((1, 1, 2), dtype=torch.float64)
    )
    world2 = environment.worlds[0]
    torch.testing.assert_close(info2["gravity_used"][0], info1["gravity_next"][0], rtol=0, atol=0)
    _assert_observation_acceleration(observations2[0], world2)

    # The measurement is allowed (and expected) to change with the physical
    # acceleration after gravity rotates. Only the normalization contract is
    # fixed; observations are not required to remain numerically identical.
    assert not torch.allclose(
        world1.episode.embodiment.measured_acceleration,
        world2.episode.embodiment.measured_acceleration,
        rtol=1e-8,
        atol=1e-8,
    )
    assert torch.all(observations1[..., 5:7].abs() <= 1.0)
    assert torch.all(observations2[..., 5:7].abs() <= 1.0)
    assert torch.all(observations1[..., 7].abs() <= 1.0)
    assert torch.all(observations2[..., 7].abs() <= 1.0)
    assert torch.all(observations1[..., 20:22].abs() <= 1.0)
    assert torch.all(observations2[..., 20:22].abs() <= 1.0)
    assert LINEAR_VELOCITY_SCALE_MPS == 12.0
    assert ANGULAR_VELOCITY_SCALE_RADPS == 4.0
