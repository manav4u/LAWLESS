
"""Mixed-world completion and frozen-world stepping contract."""

from dataclasses import fields, is_dataclass, replace

import pytest
import torch

from lawless.environment import reset, step
from test_p1_batch_equivalence import _config


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return ("tensor", value.clone())
    if is_dataclass(value):
        return (
            "dataclass",
            type(value),
            tuple(
                (f.name, _snapshot(getattr(value, f.name)))
                for f in fields(value)
            ),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_snapshot(v) for v in value))
    if isinstance(value, list):
        return ("list", tuple(_snapshot(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple((k, _snapshot(v)) for k, v in value.items()))
    return ("value", value)


def _assert_snapshot(value, snapshot):
    kind = snapshot[0]

    if kind == "tensor":
        assert isinstance(value, torch.Tensor)
        assert torch.equal(value, snapshot[1])
    elif kind == "dataclass":
        _, expected_type, children = snapshot
        assert type(value) is expected_type
        for name, child_snapshot in children:
            _assert_snapshot(getattr(value, name), child_snapshot)
    elif kind in ("tuple", "list"):
        children = snapshot[1]
        assert len(value) == len(children)
        for child, child_snapshot in zip(value, children):
            _assert_snapshot(child, child_snapshot)
    elif kind == "dict":
        children = snapshot[1]
        assert value.keys() == dict(children).keys()
        for key, child_snapshot in children:
            _assert_snapshot(value[key], child_snapshot)
    else:
        assert value == snapshot[1]


def test_completed_world_freezes_while_other_advances_then_rejects_step():
    config = _config()
    config.world.obstacle_count = 0
    config.objectives.core_count = 0
    config.validate()

    env, _, _ = reset(
        config, num_worlds=2, num_agents=1, dtype=torch.float64
    )

    worlds = []
    for wi, world in enumerate(env.worlds):
        physics = world.episode.physics
        physics = replace(
            physics,
            energy=torch.full_like(
                physics.energy, 1e-6 if wi == 0 else 1.0
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

    env = replace(env, worlds=tuple(worlds))
    actions = torch.ones((2, 1, 2), dtype=torch.float64)

    # Complete world zero through actual physical energy death.
    env, obs, _, terminated, truncated, info = step(env, actions)

    assert env.worlds[0].completed
    assert not env.worlds[1].completed
    assert not env.worlds[0].episode.physics.alive.any()
    assert info["energy_death_this_step"][0].all()
    assert info["world_completed"][0]
    assert not info["world_completed"][1]

    frozen_world = env.worlds[0]
    frozen_world_snapshot = _snapshot(frozen_world)
    frozen_obs_snapshot = _snapshot(obs[0])
    frozen_terminated_snapshot = _snapshot(terminated[0])
    frozen_truncated_snapshot = _snapshot(truncated[0])
    frozen_clock = frozen_world.step_count

    # These are the actual per-world transition-event keys in public step().
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

    # World one remains physically alive and advances until deadline truncation.
    previous_active_clock = env.worlds[1].step_count
    completed_by_deadline = False

    for _ in range(100):
        if env.all_completed:
            break

        env, obs, _, terminated, truncated, info = step(env, actions)

        # Compare every field, including physics, history, gravity and clock.
        _assert_snapshot(env.worlds[0], frozen_world_snapshot)
        _assert_snapshot(obs[0], frozen_obs_snapshot)
        _assert_snapshot(terminated[0], frozen_terminated_snapshot)
        _assert_snapshot(truncated[0], frozen_truncated_snapshot)

        assert env.worlds[0].step_count == frozen_clock
        assert info["elapsed_steps"][0].item() == frozen_clock
        assert info["elapsed_time_s"][0].item() == pytest.approx(
            frozen_clock * env.worlds[0].physics_config.dt
        )
        assert info["world_completed"][0]

        # Every transition event for the frozen world must be zero.
        for key in event_keys:
            assert not info[key][0].any(), key

        # The active world's clock must strictly advance on every transition.
        active_world = env.worlds[1]
        assert active_world.step_count > previous_active_clock
        previous_active_clock = active_world.step_count

        if active_world.completed:
            completed_by_deadline = True
            assert active_world.truncated
            assert active_world.episode.physics.alive.any()
            assert truncated[1].any()
            assert not terminated[1].any()
            assert info["physical_alive"][1].any()
            assert info["truncated"][1].any()
            assert info["world_completed"][1]
        else:
            assert not info["world_completed"][1]

    assert completed_by_deadline
    assert env.all_completed
    assert env.worlds[0].step_count == frozen_clock
    assert env.worlds[1].completed
    assert env.worlds[1].truncated
    assert env.worlds[1].episode.physics.alive.any()

    # Snapshot independently before attempting the rejected all-complete step.
    before_rejection = _snapshot(env)

    with pytest.raises(
        RuntimeError,
        match="all environment worlds are complete",
    ):
        step(env, actions)

    _assert_snapshot(env, before_rejection)
