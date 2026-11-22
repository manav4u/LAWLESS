"""Bounded CPU batch-vs-serial equivalence over the public environment API."""

import torch

from lawless.environment import EnvironmentState, reset, step
from lawless.environment.episode import WorldEpisodeState
from lawless.world.episode_state import AgentEpisodeState
from lawless.world.embodiment import AgentEmbodimentState
from lawless.world.objective_state import AgentCoreCollectionState
from lawless.physics.state import AgentState


_RTOL = 1.0e-5
_ATOL = 1.0e-6

# Ownership is derived from the public info assembly: these fields are stacked
# over the candidate dimension within each world.
_CANDIDATE_INFO_KEYS = {
    'newly_collected',
    'core_collection_count',
    'energy_restored',
    'energy_spent',
    'newly_extracted',
    'collision_contact',
    'unresolved_contact',
    'boundary_escape',
    'failure_this_step',
    'energy_death_this_step',
    'actuators_enabled',
    'physical_alive',
    'extracted',
    'failed',
    'truncated',
    'terminated',
}
_WORLD_INFO_KEYS = {
    'gravity_used',
    'gravity_next',
    'shift_applied_this_step',
    'shift_applied',
    'world_completed',
    'elapsed_steps',
    'elapsed_time_s',
}


def _config():
    from lawless.world.config import EnvironmentConfig, EpisodeConfig

    config = EnvironmentConfig(seed=710)
    config.world.width_m = 1000.0
    config.world.height_m = 1000.0
    config.world.obstacle_count = 2
    config.objectives.core_count = 2
    config.episode = EpisodeConfig(
        episode_duration_s=0.5,
        shift_time_s=0.2,
        dt_s=0.1,
    )
    config.validate()
    return config


def _slice_agent_state(physics, index):
    return AgentState(
        position=physics.position[index:index + 1].clone(),
        velocity=physics.velocity[index:index + 1].clone(),
        angle=physics.angle[index:index + 1].clone(),
        angular_velocity=physics.angular_velocity[index:index + 1].clone(),
        energy=physics.energy[index:index + 1].clone(),
        alive=physics.alive[index:index + 1].clone(),
    )


def _candidate_reference(world, candidate_index, config):
    """Construct a one-candidate reference retaining the source shared layout."""
    source = world.episode
    physics = _slice_agent_state(source.physics, candidate_index)
    embodiment = AgentEmbodimentState(
        physics=physics,
        previous_actions=source.embodiment.previous_actions[
            candidate_index:candidate_index + 1
        ].clone(),
        measured_acceleration=source.embodiment.measured_acceleration[
            candidate_index:candidate_index + 1
        ].clone(),
        extracted=source.embodiment.extracted[
            candidate_index:candidate_index + 1
        ].clone(),
        failed=source.embodiment.failed[
            candidate_index:candidate_index + 1
        ].clone(),
    )
    objectives = AgentCoreCollectionState(
        source.objectives.collected[candidate_index:candidate_index + 1].clone()
    )
    episode = AgentEpisodeState(
        generated_world=source.generated_world,
        spawn_indices=source.spawn_indices[candidate_index:candidate_index + 1].clone(),
        embodiment=embodiment,
        objectives=objectives,
    )
    reference_world = WorldEpisodeState(
        episode=episode,
        physics_config=world.physics_config,
        gravity=world.gravity.clone(),
        shift_direction=world.shift_direction,
        step_count=world.step_count,
        shift_applied=world.shift_applied,
        truncated=world.truncated[candidate_index:candidate_index + 1].clone(),
        completed=world.completed,
        world_seed=world.world_seed,
        condition_seed=world.condition_seed,
        initial_gravity=world.initial_gravity.clone(),
        linear_damping_s_inv=world.linear_damping_s_inv,
    )
    return EnvironmentState((reference_world,), config, num_agents=1)


def _assert_float(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=_RTOL, atol=_ATOL)


def _assert_value_equal(actual, expected):
    """Compare regenerated layout values without requiring object identity."""
    if isinstance(actual, torch.Tensor):
        assert isinstance(expected, torch.Tensor)
        assert actual.shape == expected.shape
        if actual.dtype == torch.bool or not actual.is_floating_point():
            assert torch.equal(actual, expected)
        else:
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return
    if isinstance(actual, dict):
        assert isinstance(expected, dict)
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_value_equal(actual[key], expected[key])
        return
    if isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_value_equal(a, b)
        return
    if hasattr(actual, '__dict__'):
        assert type(actual) is type(expected)
        assert vars(actual).keys() == vars(expected).keys()
        for key in vars(actual):
            _assert_value_equal(vars(actual)[key], vars(expected)[key])
        return
    assert actual == expected


def _assert_world_state_equal(batch_world, reference_world, *, shared_world_identity=True):
    a = batch_world.episode
    b = reference_world.episode
    assert batch_world.step_count == reference_world.step_count
    assert batch_world.shift_applied == reference_world.shift_applied
    assert batch_world.completed == reference_world.completed
    assert batch_world.shift_direction == reference_world.shift_direction
    assert batch_world.world_seed == reference_world.world_seed
    assert batch_world.condition_seed == reference_world.condition_seed
    assert batch_world.linear_damping_s_inv == reference_world.linear_damping_s_inv
    torch.testing.assert_close(batch_world.gravity, reference_world.gravity, rtol=0, atol=0)
    torch.testing.assert_close(batch_world.initial_gravity, reference_world.initial_gravity, rtol=0, atol=0)
    torch.testing.assert_close(batch_world.truncated, reference_world.truncated, rtol=0, atol=0)

    for field in (
        'position', 'velocity', 'angle', 'angular_velocity', 'energy', 'alive'
    ):
        actual = getattr(a.physics, field)
        expected = getattr(b.physics, field)
        if actual.dtype == torch.bool:
            assert torch.equal(actual, expected)
        else:
            _assert_float(actual, expected)

    for field in (
        'previous_actions', 'measured_acceleration', 'extracted', 'failed'
    ):
        actual = getattr(a.embodiment, field)
        expected = getattr(b.embodiment, field)
        if actual.dtype == torch.bool:
            assert torch.equal(actual, expected)
        else:
            _assert_float(actual, expected)

    assert torch.equal(a.spawn_indices, b.spawn_indices)
    assert torch.equal(a.objectives.collected, b.objectives.collected)
    if shared_world_identity:
        assert a.generated_world is b.generated_world
    else:
        _assert_value_equal(a.generated_world, b.generated_world)


def _assert_candidate_info_matches_batch(batch_info, candidate_info, world_index, candidate_index):
    """Compare info using explicit public-field ownership, never shape guessing."""
    assert batch_info.keys() == candidate_info.keys()
    assert set(batch_info) == _CANDIDATE_INFO_KEYS | _WORLD_INFO_KEYS

    for key, candidate_value in candidate_info.items():
        batch_value = batch_info[key]
        assert isinstance(candidate_value, torch.Tensor), key
        assert isinstance(batch_value, torch.Tensor), key

        if key in _CANDIDATE_INFO_KEYS:
            expected = batch_value[
                world_index:world_index + 1,
                candidate_index:candidate_index + 1,
            ]
        elif key in _WORLD_INFO_KEYS:
            expected = batch_value[world_index:world_index + 1]
        else:
            raise AssertionError(f'Unclassified info key: {key}')

        assert expected.shape == candidate_value.shape, key
        if candidate_value.dtype == torch.bool or not candidate_value.is_floating_point():
            assert torch.equal(candidate_value, expected), key
        else:
            _assert_float(candidate_value, expected)


def test_cpu_batched_worlds_and_candidates_match_serial_references():
    """B worlds and N candidates agree with references sliced from the same reset."""
    config = _config()
    batch, observations, reset_info = reset(
        config, num_worlds=3, num_agents=2, dtype=torch.float64
    )
    assert batch.batch_shape == (3, 2)
    assert observations.shape == (3, 2, 23)

    world_references = [
        EnvironmentState((world,), config, num_agents=2)
        for world in batch.worlds
    ]
    candidate_references = [
        [_candidate_reference(world, candidate, config) for candidate in range(2)]
        for world in batch.worlds
    ]

    assert [world.world_seed for world in batch.worlds] == reset_info['world_seeds'].tolist()
    assert [world.condition_seed for world in batch.worlds] == reset_info['condition_seeds'].tolist()
    for index, world in enumerate(batch.worlds):
        torch.testing.assert_close(
            world.initial_gravity, reset_info['initial_gravity'][index], rtol=0, atol=0
        )
        for reference in candidate_references[index]:
            assert reference.worlds[0].episode.generated_world is world.episode.generated_world

    generator = torch.Generator(device='cpu')
    generator.manual_seed(98765)
    actions_sequence = [
        torch.rand((3, 2, 2), generator=generator, dtype=torch.float64) * 2.0 - 1.0
        for _ in range(5)
    ]

    for actions in actions_sequence:
        batch, batch_obs, batch_rewards, batch_terminated, batch_truncated, batch_info = step(
            batch, actions
        )

        for world_index in range(3):
            (
                world_references[world_index],
                serial_obs,
                serial_rewards,
                serial_terminated,
                serial_truncated,
                serial_info,
            ) = step(
                world_references[world_index], actions[world_index:world_index + 1]
            )
            _assert_float(batch_obs[world_index:world_index + 1], serial_obs)
            _assert_float(batch_rewards[world_index:world_index + 1], serial_rewards)
            assert torch.equal(batch_terminated[world_index:world_index + 1], serial_terminated)
            assert torch.equal(batch_truncated[world_index:world_index + 1], serial_truncated)
            for key, value in serial_info.items():
                torch.testing.assert_close(
                    batch_info[key][world_index:world_index + 1], value, rtol=0, atol=0
                )
            _assert_world_state_equal(batch.worlds[world_index], world_references[world_index].worlds[0])

            for candidate_index in range(2):
                reference = candidate_references[world_index][candidate_index]
                (
                    reference,
                    candidate_obs,
                    candidate_rewards,
                    candidate_terminated,
                    candidate_truncated,
                    candidate_info,
                ) = step(
                    reference,
                    actions[world_index, candidate_index:candidate_index + 1].unsqueeze(0),
                )
                candidate_references[world_index][candidate_index] = reference

                _assert_float(
                    batch_obs[world_index, candidate_index:candidate_index + 1],
                    candidate_obs[0],
                )
                _assert_float(
                    batch_rewards[world_index, candidate_index:candidate_index + 1],
                    candidate_rewards[0],
                )
                assert torch.equal(
                    batch_terminated[world_index, candidate_index:candidate_index + 1],
                    candidate_terminated[0],
                )
                assert torch.equal(
                    batch_truncated[world_index, candidate_index:candidate_index + 1],
                    candidate_truncated[0],
                )
                _assert_candidate_info_matches_batch(
                    batch_info, candidate_info, world_index, candidate_index
                )
                _assert_world_state_equal(
                    _candidate_reference(batch.worlds[world_index], candidate_index, config).worlds[0],
                    reference.worlds[0],
                )

    assert batch.all_completed
    assert all(reference.all_completed for reference in world_references)
    assert all(
        reference.all_completed
        for group in candidate_references
        for reference in group
    )


def test_cpu_batched_execution_replays_identically_from_same_initial_reset():
    """Independent resets and rollouts reproduce values, not object identities."""
    config = _config()
    first, _, _ = reset(config, num_worlds=2, num_agents=2, dtype=torch.float64)
    second, _, _ = reset(config, num_worlds=2, num_agents=2, dtype=torch.float64)

    generator = torch.Generator(device='cpu')
    generator.manual_seed(1234)
    actions_sequence = [
        torch.rand((2, 2, 2), generator=generator, dtype=torch.float64) * 2.0 - 1.0
        for _ in range(5)
    ]

    for actions in actions_sequence:
        first, obs_a, rewards_a, term_a, trunc_a, info_a = step(first, actions)
        second, obs_b, rewards_b, term_b, trunc_b, info_b = step(second, actions)
        torch.testing.assert_close(obs_a, obs_b, rtol=0, atol=0)
        torch.testing.assert_close(rewards_a, rewards_b, rtol=0, atol=0)
        assert torch.equal(term_a, term_b)
        assert torch.equal(trunc_a, trunc_b)
        assert info_a.keys() == info_b.keys()
        for key in info_a:
            torch.testing.assert_close(info_a[key], info_b[key], rtol=0, atol=0)
        for world_a, world_b in zip(first.worlds, second.worlds):
            _assert_world_state_equal(
                world_a, world_b, shared_world_identity=False
            )
