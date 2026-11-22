"""Integration tests for the P1 independent-world environment composition."""

from dataclasses import replace

import pytest
import torch

from lawless.environment import EnvironmentState, WorldEpisodeState, reset, step
from lawless.world.config import EnvironmentConfig, EpisodeConfig


def _config(*, duration=4.0, shift=2.0, dt=1.0, seed=42):
    config = EnvironmentConfig(seed=seed)
    config.world.obstacle_count = 2
    config.objectives.core_count = 2
    config.episode = EpisodeConfig(
        episode_duration_s=duration,
        shift_time_s=shift,
        dt_s=dt,
    )
    config.validate()
    return config


def test_reset_builds_independent_worlds_with_one_shared_spawn_per_world():
    env, observations, info = reset(_config(), 3, 5)

    assert observations.shape == (3, 5, 23)
    assert info["spawn_indices"].shape == (3, 5)
    assert torch.equal(info["spawn_indices"], torch.zeros((3, 5), dtype=torch.int64))
    assert env.worlds[0].episode.generated_world is not env.worlds[1].episode.generated_world
    for world_index, world in enumerate(env.worlds):
        assert world.episode.generated_world.spawn_positions.shape == (1, 2)
        assert torch.equal(world.episode.spawn_indices, torch.zeros(5, dtype=torch.int64))
        assert world.episode.physics.position.shape == (5, 2)
        assert torch.equal(
            world.episode.physics.position,
            world.episode.physics.position[0:1].expand(5, 2),
        )
        assert torch.equal(
            observations[world_index, 0],
            observations[world_index, 1],
        )












def test_shift_is_applied_after_boundary_transition_and_diagnostics_distinguish_gravity():
    env, _, _ = reset(_config(duration=4.0, shift=2.0, dt=1.0), 1, 1)
    initial = env.worlds[0].gravity.clone()
    env, _, _, _, _, info1 = step(env, torch.zeros((1, 1, 2)))
    assert torch.equal(info1["gravity_used"][0], initial)
    assert torch.equal(info1["gravity_next"][0], initial)
    assert not bool(info1["shift_applied_this_step"][0])

    env, _, _, _, _, info2 = step(env, torch.zeros((1, 1, 2)))
    rotated = info2["gravity_next"][0]
    assert torch.equal(info2["gravity_used"][0], initial)
    assert torch.equal(rotated, env.worlds[0].gravity)
    assert bool(info2["shift_applied_this_step"][0])
    assert bool(info2["shift_applied"][0])

    env, _, _, _, _, info3 = step(env, torch.zeros((1, 1, 2)))
    assert torch.equal(info3["gravity_used"][0], rotated)
    assert torch.equal(info3["gravity_next"][0], rotated)
    assert not bool(info3["shift_applied_this_step"][0])


def test_finite_out_of_range_actions_are_clipped_and_recorded_after_clipping():
    env, _, _ = reset(_config(), 1, 2)
    actions = torch.tensor([[[3.0, -4.0], [-2.0, 5.0]]])
    env, _, _, _, _, _ = step(env, actions)
    assert torch.equal(
        env.worlds[0].episode.embodiment.previous_actions,
        torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_action_rejected_before_any_world_advances(bad):
    env, _, _ = reset(_config(), 2, 2)
    actions = torch.zeros((2, 2, 2))
    actions[1, 1, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        step(env, actions)
    assert [world.step_count for world in env.worlds] == [0, 0]


def test_wrong_action_shape_rejected_before_any_world_advances():
    env, _, _ = reset(_config(), 2, 2)
    with pytest.raises(ValueError, match="shape"):
        step(env, torch.zeros((2, 1, 2)))
    assert [world.step_count for world in env.worlds] == [0, 0]


def test_completed_world_is_frozen_while_other_world_continues():
    env, _, _ = reset(_config(duration=5.0, shift=3.0, dt=1.0), 2, 2)
    worlds = list(env.worlds)
    first = worlds[0]
    dead_physics = first.episode.physics.clone()
    dead_physics.alive[:] = False
    dead_episode = first.episode.replace_physics(dead_physics)
    worlds[0] = replace(first, episode=dead_episode, completed=True)
    env = EnvironmentState(tuple(worlds), env.config, env.num_agents)

    old_first = env.worlds[0]
    old_first_position = old_first.episode.physics.position.clone()
    env_next, observations, rewards, terminated, truncated, info = step(
        env, torch.zeros((2, 2, 2))
    )

    assert env_next.worlds[0] is old_first
    assert torch.equal(env_next.worlds[0].episode.physics.position, old_first_position)
    assert env_next.worlds[0].step_count == old_first.step_count
    assert env_next.worlds[1].step_count == 1
    assert terminated[0].all()
    assert not truncated[0].any()
    assert info["world_completed"].tolist() == [True, False]


def test_deadline_truncates_live_agents_without_changing_physical_alive_mask():
    env, _, _ = reset(_config(duration=2.0, shift=1.0, dt=1.0), 1, 3)
    env, _, _, term1, trunc1, _ = step(env, torch.zeros((1, 3, 2)))
    assert not term1.any() and not trunc1.any()
    env, _, _, term2, trunc2, info = step(env, torch.zeros((1, 3, 2)))
    assert not term2.any()
    assert trunc2.all()
    assert env.worlds[0].completed
    assert env.worlds[0].episode.physics.alive.all()
    assert not env.worlds[0].episode.embodiment.extracted.any()
    assert not env.worlds[0].episode.embodiment.failed.any()
    assert info["elapsed_steps"].tolist() == [2]


def test_all_finished_environment_rejects_another_step_without_mutation():
    env, _, _ = reset(_config(duration=1.0, shift=0.0, dt=1.0), 1, 2)
    env, _, _, _, truncated, _ = step(env, torch.zeros((1, 2, 2)))
    assert truncated.all()
    before = env.worlds[0].episode.physics.position.clone()
    with pytest.raises(RuntimeError, match="complete"):
        step(env, torch.zeros((1, 2, 2)))
    assert torch.equal(env.worlds[0].episode.physics.position, before)


def test_extraction_success_is_the_only_scalar_reward_and_is_raw():
    config = _config(duration=1.0, shift=0.0, dt=1.0)
    env, _, _ = reset(config, 1, 2)
    world = env.worlds[0]
    physics = world.episode.physics.clone()
    physics.position[:] = world.episode.generated_world.extraction_center
    physics.velocity.zero_()
    episode = world.episode.replace_physics(physics)
    worlds = (replace(world, episode=episode, gravity=torch.zeros_like(world.gravity)),)
    env = EnvironmentState(worlds, env.config, env.num_agents)

    env, _, rewards, terminated, truncated, info = step(
        env, torch.zeros((1, 2, 2))
    )
    assert torch.equal(rewards, torch.ones((1, 2)))
    assert info["newly_extracted"].all()
    assert terminated.all()
    assert not truncated.any()
    assert info["core_collection_count"].shape == (1, 2)
    assert info["energy_restored"].shape == (1, 2)
    assert info["failure_this_step"].shape == (1, 2)


def test_core_collection_state_is_independent_per_candidate():
    env, _, _ = reset(_config(), 1, 3)
    episode = env.worlds[0].episode
    updated, newly = episode.mark_collected(
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
    )
    assert newly.tolist() == [True]
    assert updated.objectives.collected[:, 0].tolist() == [False, True, False]
    assert episode.objectives.collected[:, 0].tolist() == [False, False, False]


def test_naturally_completed_world_freezes_while_other_world_advances():
    env, _, _ = reset(_config(duration=5.0, shift=3.0, dt=1.0), 2, 2)
    worlds = list(env.worlds)
    first = worlds[0]
    zero_energy_physics = first.episode.physics.clone()
    zero_energy_physics.energy.zero_()
    zero_energy_episode = first.episode.replace_physics(zero_energy_physics)
    worlds[0] = replace(first, episode=zero_energy_episode)
    env = EnvironmentState(tuple(worlds), env.config, env.num_agents)

    env, observations1, _, terminated1, truncated1, info1 = step(
        env, torch.zeros((2, 2, 2))
    )
    assert terminated1[0].all()
    assert not truncated1[0].any()
    assert env.worlds[0].completed
    assert env.worlds[0].step_count == 1
    assert not env.worlds[1].completed
    frozen_state = env.worlds[0]
    frozen_observation = observations1[0].clone()

    env, observations2, _, terminated2, truncated2, info2 = step(
        env, torch.zeros((2, 2, 2))
    )
    assert env.worlds[0] is frozen_state
    assert torch.equal(observations2[0], frozen_observation)
    assert torch.equal(terminated2[0], terminated1[0])
    assert torch.equal(truncated2[0], truncated1[0])
    assert env.worlds[0].step_count == 1
    assert env.worlds[1].step_count == 2
    assert info2["elapsed_steps"].tolist() == [1, 2]


def test_nonbinary_timestep_deadline_is_inclusive_without_float_accumulation():
    config = _config(duration=0.3, shift=0.1, dt=0.1)
    env, _, _ = reset(config, 1, 1)
    world = env.worlds[0]
    world = replace(world, gravity=torch.zeros_like(world.gravity))
    env = EnvironmentState((world,), env.config, env.num_agents)

    env, _, _, _, _, _ = step(env, torch.zeros((1, 1, 2)))
    env, _, _, _, _, _ = step(env, torch.zeros((1, 1, 2)))
    assert not env.worlds[0].completed

    world = env.worlds[0]
    physics = world.episode.physics.clone()
    physics.position[:] = world.episode.generated_world.extraction_center
    world = replace(world, episode=world.episode.replace_physics(physics))
    env = EnvironmentState((world,), env.config, env.num_agents)

    env, _, rewards, terminated, truncated, info = step(
        env, torch.zeros((1, 1, 2))
    )
    assert rewards.item() == 1.0
    assert terminated.item()
    assert not truncated.item()
    assert info["elapsed_time_s"].item() == config.episode.episode_duration_s


def test_zero_time_shift_is_applied_at_reset_before_transition_one():
    from lawless.physics.dynamics import rotate_gravity_90

    config = _config(duration=2.0, shift=0.0, dt=1.0)
    env, _, info = reset(config, 1, 1)
    world = env.worlds[0]
    expected = rotate_gravity_90(world.initial_gravity, world.shift_direction)
    assert info["shift_applied_at_reset"].tolist() == [True]
    assert torch.equal(world.gravity, expected)
    env, _, _, _, _, step_info = step(env, torch.zeros((1, 1, 2)))
    assert torch.equal(step_info["gravity_used"][0], expected)
    assert bool(step_info["shift_applied"][0])
    assert not bool(step_info["shift_applied_this_step"][0])
