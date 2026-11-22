import torch

from lawless.physics.constants import PhysicsConfig
from lawless.physics.state import make_state
from lawless.physics.simulator import step


def _run():
    cfg = PhysicsConfig()
    state = make_state((2, 4), dtype=torch.float64)
    actions = torch.full((2, 4, 2), 0.25, dtype=torch.float64)
    gravity = torch.tensor([0.0, 4.0], dtype=torch.float64)
    for _ in range(100):
        state, _ = step(state, actions, gravity, cfg)
    return state


def test_repeated_cpu_runs_are_identical():
    a, b = _run(), _run()
    for field in a.__dataclass_fields__:
        assert torch.equal(getattr(a, field), getattr(b, field))
