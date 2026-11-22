import torch

from lawless.physics.constants import PhysicsConfig
from lawless.physics.state import make_state
from lawless.physics.simulator import step


def test_batched_equals_independent_agents():
    cfg = PhysicsConfig()
    n = 7
    batched = make_state(n, dtype=torch.float64)
    batched.position[:, 0] = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    batched.velocity[:, 1] = torch.linspace(-1.0, 1.0, n, dtype=torch.float64)
    batched.angle[:, 0] = torch.linspace(-0.5, 0.5, n, dtype=torch.float64)
    actions = torch.rand((n, 2), generator=torch.Generator().manual_seed(7), dtype=torch.float64) * 2 - 1
    gravity = torch.tensor([0.0, 4.0], dtype=torch.float64)
    out_batch, _ = step(batched, actions, gravity, cfg)

    outputs = []
    for i in range(n):
        one = make_state(1, dtype=torch.float64)
        for field in ("position", "velocity", "angle", "angular_velocity", "energy", "alive"):
            getattr(one, field)[0] = getattr(batched, field)[i]
        out, _ = step(one, actions[i:i+1], gravity, cfg)
        outputs.append(out)

    for field in ("position", "velocity", "angle", "angular_velocity", "energy", "alive"):
        expected = torch.cat([getattr(x, field) for x in outputs], dim=0)
        torch.testing.assert_close(getattr(out_batch, field), expected)
