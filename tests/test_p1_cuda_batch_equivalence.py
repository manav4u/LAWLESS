"""CUDA batched-world and batched-agent equivalence tests.

These tests exercise the public reset/step API against independent serial
references. They do not modify production code.
"""

from dataclasses import replace

import pytest
import torch

from lawless.environment import EnvironmentState, reset, step
from test_p1_batch_equivalence import (
    _assert_candidate_info_matches_batch,
    _assert_world_state_equal,
    _candidate_reference,
    _config,
)
from test_p1_mixed_completion_freeze import _assert_snapshot, _snapshot


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for this equivalence test",
)

_RTOL = 1e-5
_ATOL = 1e-6


def _assert_tensor_equal(actual, expected):
    assert isinstance(actual, torch.Tensor)
    assert isinstance(expected, torch.Tensor)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype

    if actual.is_floating_point() or actual.is_complex():
        assert bool(torch.isfinite(actual).all().item()), (
            "actual tensor contains non-finite values"
        )
        assert bool(torch.isfinite(expected).all().item()), (
            "expected tensor contains non-finite values"
        )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=_RTOL,
            atol=_ATOL,
            equal_nan=False,
        )
    else:
        assert torch.equal(actual, expected)


def _assert_world_info_matches_batch(batch_info, serial_info, world_index):
    assert batch_info.keys() == serial_info.keys()

    for key, serial_value in serial_info.items():
        batch_value = batch_info[key]
        assert isinstance(batch_value, torch.Tensor), (
            f"info[{key!r}] must be a tensor"
        )
        _assert_tensor_equal(batch_value[world_index], serial_value[0])


def _assert_world_outputs_match(
    batch_observations,
    batch_rewards,
    batch_terminated,
    batch_truncated,
    serial_observations,
    serial_rewards,
    serial_terminated,
    serial_truncated,
    world_index,
):
    _assert_tensor_equal(
        batch_observations[world_index],
        serial_observations[0],
    )
    _assert_tensor_equal(batch_rewards[world_index], serial_rewards[0])
    _assert_tensor_equal(batch_terminated[world_index], serial_terminated[0])
    _assert_tensor_equal(batch_truncated[world_index], serial_truncated[0])


def _assert_candidate_outputs_match(
    batch_observations,
    batch_rewards,
    batch_terminated,
    batch_truncated,
    candidate_observations,
    candidate_rewards,
    candidate_terminated,
    candidate_truncated,
    world_index,
    candidate_index,
):
    _assert_tensor_equal(
        batch_observations[world_index, candidate_index],
        candidate_observations[0, 0],
    )
    _assert_tensor_equal(
        batch_rewards[world_index, candidate_index],
        candidate_rewards[0, 0],
    )
    _assert_tensor_equal(
        batch_terminated[world_index, candidate_index],
        candidate_terminated[0, 0],
    )
    _assert_tensor_equal(
        batch_truncated[world_index, candidate_index],
        candidate_truncated[0, 0],
    )


def _cuda_reset(config, *, num_worlds, num_agents):
    environment, observations, info = reset(
        config,
        num_worlds=num_worlds,
        num_agents=num_agents,
        device="cuda",
        dtype=torch.float32,
    )

    assert observations.device.type == "cuda"
    assert observations.shape == (num_worlds, num_agents, 23)
    assert info["world_seeds"].device.type == "cuda"
    return environment, observations, info


def test_cuda_batched_worlds_and_agents_match_serial_references():
    config = _config()
    num_worlds = 2
    num_agents = 2

    batch, initial_observations, _ = _cuda_reset(
        config,
        num_worlds=num_worlds,
        num_agents=num_agents,
    )
    assert initial_observations.shape == (2, 2, 23)

    # Independent one-world references and one-candidate references start
    # from the exact reset worlds used by the batched execution.
    world_references = [
        EnvironmentState(
            worlds=(batch.worlds[world_index],),
            config=config,
            num_agents=num_agents,
        )
        for world_index in range(num_worlds)
    ]
    candidate_references = [
        [
            _candidate_reference(
                batch.worlds[world_index],
                candidate_index,
                config,
            )
            for candidate_index in range(num_agents)
        ]
        for world_index in range(num_worlds)
    ]

    # Generate deterministic actions on CPU, then transfer those exact
    # values to CUDA.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(98765)
    cpu_actions = [
        torch.rand(
            (num_worlds, num_agents, 2),
            generator=generator,
            dtype=torch.float32,
        )
        for _ in range(5)
    ]

    observed_gravity_shift = False

    for cpu_action in cpu_actions:
        actions = cpu_action.to(device="cuda")

        (
            batch,
            batch_observations,
            batch_rewards,
            batch_terminated,
            batch_truncated,
            batch_info,
        ) = step(batch, actions)

        observed_gravity_shift = observed_gravity_shift or bool(
            batch_info["shift_applied_this_step"].any().item()
        )

        # Compare each batched world with its independent one-world run.
        for world_index in range(num_worlds):
            (
                world_references[world_index],
                serial_observations,
                serial_rewards,
                serial_terminated,
                serial_truncated,
                serial_info,
            ) = step(
                world_references[world_index],
                actions[world_index : world_index + 1],
            )

            _assert_world_outputs_match(
                batch_observations,
                batch_rewards,
                batch_terminated,
                batch_truncated,
                serial_observations,
                serial_rewards,
                serial_terminated,
                serial_truncated,
                world_index,
            )
            _assert_world_info_matches_batch(
                batch_info,
                serial_info,
                world_index,
            )
            _assert_world_state_equal(
                batch.worlds[world_index],
                world_references[world_index].worlds[0],
            )

        # Compare each batched candidate with an independent one-candidate run.
        for world_index in range(num_worlds):
            for candidate_index in range(num_agents):
                (
                    candidate_references[world_index][candidate_index],
                    candidate_observations,
                    candidate_rewards,
                    candidate_terminated,
                    candidate_truncated,
                    candidate_info,
                ) = step(
                    candidate_references[world_index][candidate_index],
                    actions[
                        world_index : world_index + 1,
                        candidate_index : candidate_index + 1,
                        :,
                    ],
                )

                _assert_candidate_outputs_match(
                    batch_observations,
                    batch_rewards,
                    batch_terminated,
                    batch_truncated,
                    candidate_observations,
                    candidate_rewards,
                    candidate_terminated,
                    candidate_truncated,
                    world_index,
                    candidate_index,
                )
                _assert_candidate_info_matches_batch(
                    batch_info,
                    candidate_info,
                    world_index,
                    candidate_index,
                )

                batch_candidate_reference = _candidate_reference(
                    batch.worlds[world_index],
                    candidate_index,
                    config,
                )
                _assert_world_state_equal(
                    batch_candidate_reference.worlds[0],
                    candidate_references[world_index][candidate_index].worlds[0],
                )

    assert observed_gravity_shift, "fixture did not exercise the gravity shift"
    assert batch.all_completed
    assert all(reference.all_completed for reference in world_references)
    assert all(
        reference.all_completed
        for world_candidates in candidate_references
        for reference in world_candidates
    )


def test_cuda_mixed_completion_freezes_one_world_while_other_advances():
    config = _config()
    config.world.obstacle_count = 0
    config.objectives.core_count = 0
    config.validate()

    batch, _, _ = _cuda_reset(
        config,
        num_worlds=2,
        num_agents=2,
    )

    # Cause world 0 to die from energy depletion on its first transition;
    # world 1 retains enough energy to reach the episode deadline.
    worlds = []
    for world_index, world in enumerate(batch.worlds):
        physics = world.episode.physics
        physics = replace(
            physics,
            energy=torch.full_like(
                physics.energy,
                1e-6 if world_index == 0 else 1.0,
            ),
        )
        episode = world.episode.replace_physics(physics)
        gravity = torch.zeros_like(world.gravity)
        worlds.append(
            replace(
                world,
                episode=episode,
                gravity=gravity,
                initial_gravity=gravity.clone(),
            )
        )

    batch = replace(batch, worlds=tuple(worlds))

    # The serial active-world reference starts from the same pre-step state
    # as the batched environment.
    serial_active = EnvironmentState(
        worlds=(batch.worlds[1],),
        config=config,
        num_agents=2,
    )
    actions = torch.ones(
        (2, 2, 2),
        device="cuda",
        dtype=torch.float32,
    )

    # Advance both implementations through transition one before the
    # freeze-check loop. Their transition counts therefore remain aligned.
    (
        batch,
        batch_observations,
        batch_rewards,
        batch_terminated,
        batch_truncated,
        batch_info,
    ) = step(batch, actions)

    (
        serial_active,
        serial_observations,
        serial_rewards,
        serial_terminated,
        serial_truncated,
        serial_info,
    ) = step(serial_active, actions[1:2])

    _assert_world_outputs_match(
        batch_observations,
        batch_rewards,
        batch_terminated,
        batch_truncated,
        serial_observations,
        serial_rewards,
        serial_terminated,
        serial_truncated,
        1,
    )
    _assert_world_info_matches_batch(batch_info, serial_info, 1)
    _assert_world_state_equal(
        batch.worlds[1],
        serial_active.worlds[0],
    )

    assert batch.worlds[0].completed
    assert not batch.worlds[1].completed
    assert not bool(batch.worlds[0].episode.physics.alive.any().item())
    assert bool(batch_info["energy_death_this_step"][0].all().item())
    assert bool(batch_info["world_completed"][0].item())
    assert not bool(batch_info["world_completed"][1].item())

    frozen_world = batch.worlds[0]
    frozen_world_snapshot = _snapshot(frozen_world)
    frozen_observations_snapshot = _snapshot(batch_observations[0])
    frozen_terminated_snapshot = _snapshot(batch_terminated[0])
    frozen_truncated_snapshot = _snapshot(batch_truncated[0])
    frozen_clock = frozen_world.step_count

    event_keys = (
        "newly_collected",
        "energy_restored",
        "energy_spent",
        "newly_extracted",
        "collision_contact",
        "unresolved_contact",
        "boundary_escape",
        "failure_this_step",
        "energy_death_this_step",
        "actuators_enabled",
        "shift_applied_this_step",
    )

    previous_active_clock = batch.worlds[1].step_count
    completed_by_deadline = False

    for _ in range(100):
        if batch.all_completed:
            break

        (
            batch,
            batch_observations,
            batch_rewards,
            batch_terminated,
            batch_truncated,
            batch_info,
        ) = step(batch, actions)

        (
            serial_active,
            serial_observations,
            serial_rewards,
            serial_terminated,
            serial_truncated,
            serial_info,
        ) = step(serial_active, actions[1:2])

        # Recursive independent snapshots catch in-place mutation that
        # object identity alone would not detect.
        assert batch.worlds[0] is frozen_world
        _assert_snapshot(batch.worlds[0], frozen_world_snapshot)
        _assert_snapshot(
            batch_observations[0],
            frozen_observations_snapshot,
        )
        _assert_snapshot(
            batch_terminated[0],
            frozen_terminated_snapshot,
        )
        _assert_snapshot(
            batch_truncated[0],
            frozen_truncated_snapshot,
        )

        assert batch.worlds[0].step_count == frozen_clock
        assert batch_info["elapsed_steps"][0].item() == frozen_clock
        assert batch_info["elapsed_time_s"][0].item() == pytest.approx(
            frozen_clock * batch.worlds[0].physics_config.dt
        )
        assert bool(batch_info["world_completed"][0].item())

        # Only per-transition event keys are expected to be zero for the
        # frozen world. Persistent completion and elapsed-clock info remain
        # meaningful and are checked separately above.
        for key in event_keys:
            assert not bool(batch_info[key][0].any().item()), key

        # The active world must advance and continue to match its serial run.
        active_world = batch.worlds[1]
        assert active_world.step_count > previous_active_clock
        previous_active_clock = active_world.step_count

        _assert_world_outputs_match(
            batch_observations,
            batch_rewards,
            batch_terminated,
            batch_truncated,
            serial_observations,
            serial_rewards,
            serial_terminated,
            serial_truncated,
            1,
        )
        _assert_world_info_matches_batch(batch_info, serial_info, 1)
        _assert_world_state_equal(
            batch.worlds[1],
            serial_active.worlds[0],
        )

        if active_world.completed:
            completed_by_deadline = True
            assert bool(active_world.truncated.all().item())
            assert bool(active_world.episode.physics.alive.all().item())
            assert bool(batch_truncated[1].all().item())
            assert not bool(batch_terminated[1].any().item())
            assert bool(batch_info["physical_alive"][1].all().item())
            assert bool(batch_info["truncated"][1].all().item())
            assert bool(batch_info["world_completed"][1].item())
        else:
            assert not bool(batch_info["world_completed"][1].item())

    assert completed_by_deadline
    assert batch.all_completed
    assert batch.worlds[0].step_count == frozen_clock
    assert batch.worlds[1].completed
    assert bool(batch.worlds[1].truncated.all().item())
    assert bool(batch.worlds[1].episode.physics.alive.all().item())

    # Check the final active-world masks explicitly for both agents.
    expected_truncated = torch.ones_like(
        batch_truncated[1],
        dtype=torch.bool,
    )
    expected_terminated = torch.zeros_like(
        batch_terminated[1],
        dtype=torch.bool,
    )
    expected_alive = torch.ones_like(
        batch_info["physical_alive"][1],
        dtype=torch.bool,
    )

    assert torch.equal(batch_truncated[1].to(dtype=torch.bool), expected_truncated)
    assert torch.equal(
        batch_terminated[1].to(dtype=torch.bool),
        expected_terminated,
    )
    assert torch.equal(
        batch_info["physical_alive"][1].to(dtype=torch.bool),
        expected_alive,
    )

    # A rejected all-complete step must not mutate any part of the environment.
    before_rejection = _snapshot(batch)

    with pytest.raises(
        RuntimeError,
        match="all environment worlds are complete",
    ):
        step(batch, actions)

    _assert_snapshot(batch, before_rejection)