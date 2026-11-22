"""Seed isolation and paired-candidate equivariance integration tests."""

import torch

from lawless.environment import (
    EnvironmentState,
    derive_condition_seed,
    derive_world_seed,
    reset,
    step,
)
from lawless.world.config import EnvironmentConfig, EpisodeConfig


def _config(*, seed=42):
    config = EnvironmentConfig(seed=seed)
    config.world.obstacle_count = 2
    config.objectives.core_count = 2
    config.episode = EpisodeConfig(
        episode_duration_s=4.0,
        shift_time_s=2.0,
        dt_s=1.0,
    )
    config.validate()
    return config


def test_root_seed_world_index_and_conditions_are_reproducible():
    config = _config(seed=918)
    first, obs_a, info_a = reset(config, 3, 2)
    second, obs_b, info_b = reset(config, 3, 2)

    assert torch.equal(obs_a, obs_b)
    assert torch.equal(info_a["world_seeds"], info_b["world_seeds"])
    assert torch.equal(info_a["condition_seeds"], info_b["condition_seeds"])
    assert torch.equal(info_a["initial_gravity"], info_b["initial_gravity"])
    assert torch.equal(info_a["linear_damping_s_inv"], info_b["linear_damping_s_inv"])
    assert torch.equal(info_a["shift_direction"], info_b["shift_direction"])
    for left, right in zip(first.worlds, second.worlds):
        a, b = left.episode.generated_world, right.episode.generated_world
        assert torch.equal(a.world.obstacles.centers, b.world.obstacles.centers)
        assert torch.equal(a.world.obstacles.radii, b.world.obstacles.radii)
        assert torch.equal(a.spawn_positions, b.spawn_positions)
        assert torch.equal(a.extraction_center, b.extraction_center)
        assert torch.equal(a.core_positions, b.core_positions)

def test_layout_and_conditions_do_not_depend_on_candidate_count():
    config = _config(seed=1234)
    one, _, info_one = reset(config, 2, 1)
    many, _, info_many = reset(config, 2, 7)

    assert torch.equal(info_one["world_seeds"], info_many["world_seeds"])
    assert torch.equal(info_one["condition_seeds"], info_many["condition_seeds"])
    assert torch.equal(info_one["initial_gravity"], info_many["initial_gravity"])
    assert torch.equal(info_one["linear_damping_s_inv"], info_many["linear_damping_s_inv"])
    assert torch.equal(info_one["shift_direction"], info_many["shift_direction"])
    for a_world, b_world in zip(one.worlds, many.worlds):
        a, b = a_world.episode.generated_world, b_world.episode.generated_world
        assert torch.equal(a.world.obstacles.centers, b.world.obstacles.centers)
        assert torch.equal(a.spawn_positions, b.spawn_positions)
        assert torch.equal(a.extraction_center, b.extraction_center)
        assert torch.equal(a.core_positions, b.core_positions)

def test_world_streams_are_separate_and_global_rng_is_untouched():
    config = _config(seed=73)
    torch.manual_seed(1001)
    expected = torch.rand(5)
    torch.manual_seed(1001)
    before = torch.random.get_rng_state().clone()
    env, _, info = reset(config, 4, 2)
    after = torch.random.get_rng_state()

    assert torch.equal(before, after)
    assert torch.equal(torch.rand(5), expected)
    assert len(set(info["world_seeds"].tolist())) == 4
    assert len(set(info["condition_seeds"].tolist())) == 4
    assert not torch.equal(info["world_seeds"], info["condition_seeds"])

def test_identical_candidates_have_identical_initial_observations_and_trajectories():
    env, observations, _ = reset(_config(), 1, 4)
    assert torch.equal(observations[:, 0], observations[:, 1])
    actions = torch.tensor([[[0.2, -0.1]] * 4])
    for _ in range(3):
        env, observations, rewards, terminated, truncated, info = step(env, actions)
        assert torch.equal(observations[:, 0], observations[:, 1])
        assert torch.equal(
            env.worlds[0].episode.physics.position[0],
            env.worlds[0].episode.physics.position[1],
        )
        assert torch.equal(
            env.worlds[0].episode.objectives.collected[0],
            env.worlds[0].episode.objectives.collected[1],
        )

def test_candidate_permutation_only_permutes_candidate_outcomes():
    config = _config(seed=88)
    env_a, _, _ = reset(config, 1, 4)
    env_b, _, _ = reset(config, 1, 4)
    actions = torch.tensor([[
        [0.8, 0.1],
        [-0.4, 0.2],
        [0.0, -0.7],
        [0.3, 0.0],
    ]])
    permutation = torch.tensor([2, 0, 3, 1])
    next_a = step(env_a, actions)
    next_b = step(env_b, actions[:, permutation])

    state_a, obs_a, reward_a, term_a, trunc_a, info_a = next_a
    state_b, obs_b, reward_b, term_b, trunc_b, info_b = next_b
    inverse = torch.argsort(permutation)
    assert torch.allclose(obs_a, obs_b[:, inverse], rtol=0, atol=0)
    assert torch.equal(reward_a, reward_b[:, inverse])
    assert torch.equal(term_a, term_b[:, inverse])
    assert torch.equal(trunc_a, trunc_b[:, inverse])
    assert torch.allclose(
        state_a.worlds[0].episode.physics.position,
        state_b.worlds[0].episode.physics.position[inverse],
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        info_a["newly_collected"],
        info_b["newly_collected"][:, inverse],
    )

def test_seed_derivation_has_stable_world_and_condition_stream_vectors():
    assert [derive_world_seed(42, i) for i in range(4)] == [
        8402223576077616011,
        402112292510538952,
        3945394111967724371,
        1189284562730336803,
    ]
    assert [derive_condition_seed(42, i) for i in range(4)] == [
        358601024797680128,
        1563171651273388081,
        5750680954760213167,
        2139727023731356246,
    ]
