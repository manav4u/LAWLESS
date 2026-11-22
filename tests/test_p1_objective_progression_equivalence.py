"""Public-path objective progression batch equivalence."""
from dataclasses import replace

import torch

from lawless.environment import reset, step
from tests.test_p1_batch_equivalence import (
    _assert_world_state_equal,
    _assert_candidate_info_matches_batch,
    _candidate_reference,
    _config,
)


def test_cpu_objective_progression_matches_serial_candidates():
    config = _config()
    config.world.obstacle_count = 0
    config.validate()

    env, _, _ = reset(config, num_worlds=2, num_agents=2, dtype=torch.float64)
    worlds = []
    for w in env.worlds:
        p = w.episode.physics
        pos = w.episode.generated_world.core_positions[:2].clone()
        p = replace(p, position=pos, velocity=torch.zeros_like(p.velocity),
                    angular_velocity=torch.zeros_like(p.angular_velocity),
                    energy=torch.full_like(p.energy, 1.0))
        ep = w.episode.replace_physics(p)
        g = torch.zeros_like(w.gravity)
        worlds.append(replace(w, episode=ep, gravity=g, initial_gravity=g.clone()))
    env = replace(env, worlds=tuple(worlds))

    refs = [[_candidate_reference(w, i, config) for i in range(2)]
            for w in env.worlds]
    actions = torch.zeros((2, 2, 2), dtype=torch.float64)

    for transition in range(4):
        env, _, _, _, _, info = step(env, actions)
        serial_infos = [[None, None], [None, None]]
        for wi in range(2):
            for ci in range(2):
                r = refs[wi][ci]
                r, _, _, _, _, si = step(r, actions[wi, ci:ci+1].unsqueeze(0))
                refs[wi][ci] = r
                serial_infos[wi][ci] = si
                _assert_candidate_info_matches_batch(info, si, wi, ci)
                sliced = _candidate_reference(env.worlds[wi], ci, config)
                _assert_world_state_equal(sliced.worlds[0], r.worlds[0],
                                          shared_world_identity=True)

        history = [w.episode.objectives.collected for w in env.worlds]
        if transition == 0:
            assert info["newly_collected"].any()
            assert torch.all(info["energy_restored"] > 0)
            assert torch.all(info["core_collection_count"] >= 1)
            for wi, h in enumerate(history):
                assert not torch.equal(h[0], h[1])
                assert torch.equal(h, info["newly_collected"][wi])
                assert torch.equal(
                    info["core_collection_count"][wi],
                    info["newly_collected"][wi].sum(dim=-1),
                )
        else:
            assert not info["newly_collected"].any()
            assert not info["energy_restored"].any()
            assert not info["core_collection_count"].any()
            for wi in range(2):
                assert torch.equal(history[wi], prior[wi])

        if transition == 0:
            prior = [h.clone() for h in history]