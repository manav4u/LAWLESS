import pytest
import torch

from lawless.physics.constants import PhysicsConfig
from lawless.physics.state import make_state
from lawless.physics.simulator import step


def test_simulator_first_step_reference():
    cfg = PhysicsConfig(linear_damping=0.0, angular_damping=0.0, thrust_power=0.0, torque_power=0.0)
    state = make_state(1, dtype=torch.float64)
    actions = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    gravity = torch.zeros((2,), dtype=torch.float64)
    new_state, _ = step(state, actions, gravity, cfg)
    torch.testing.assert_close(new_state.velocity, torch.tensor([[0.2, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(new_state.position, torch.tensor([[0.2 / 60.0, 0.0]], dtype=torch.float64))


def test_exhausted_agent_actuators_disabled():
    cfg = PhysicsConfig(dt=1.0, initial_energy=0.1, max_energy=100.0,
                       thrust_power=2.0, torque_power=0.0, linear_damping=0.0)
    state = make_state(1, dtype=torch.float64, energy=0.1)
    action = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    new_state, _ = step(state, action, torch.zeros(2, dtype=torch.float64), cfg)
    assert new_state.energy.item() == 0.0
    # Exhaustion is detected from energy at step start; this step is permitted,
    # and the next step has actuators disabled. This boundary convention is explicit.
    next_state, _ = step(new_state, action, torch.zeros(2, dtype=torch.float64), cfg)
    assert torch.equal(next_state.velocity, new_state.velocity)


def test_dead_agent_is_fully_frozen():
    cfg = PhysicsConfig(
        dt=0.1,
        max_energy=100.0,
        thrust_power=10.0,
        torque_power=5.0,
        linear_damping=0.15,
        angular_damping=0.2,
    )

    state = make_state(1, dtype=torch.float64, energy=50.0)
    state.position[:] = torch.tensor([[3.0, 4.0]], dtype=torch.float64)
    state.velocity[:] = torch.tensor([[2.0, -1.0]], dtype=torch.float64)
    state.angle[:] = torch.tensor([[0.7]], dtype=torch.float64)
    state.angular_velocity[:] = torch.tensor([[1.2]], dtype=torch.float64)
    state.alive[:] = False

    action = torch.tensor([[1.0, 1.0]], dtype=torch.float64)
    gravity = torch.tensor([0.0, 4.0], dtype=torch.float64)

    before = state.clone()
    after, diagnostics = step(state, action, gravity, cfg)

    for field in (
        "position",
        "velocity",
        "angle",
        "angular_velocity",
        "energy",
        "alive",
    ):
        torch.testing.assert_close(
            getattr(after, field),
            getattr(before, field),
            msg=f"Dead agent changed field: {field}",
        )


def test_energy_boundary_full_step_actuation_contract():
    cfg = PhysicsConfig(thrust_power=1.5, torque_power=0.5, dt=1/60)
    g = torch.zeros(2)
    a = torch.tensor([[1.0, 0.0]])
    s = make_state(1, energy=0.0)
    n, d = step(s, a, g, cfg)
    assert not d["actuators_enabled"].any()
    assert torch.equal(n.velocity, s.velocity)
    assert n.energy.item() == 0.0
    s = make_state(1, energy=0.025)
    n, d = step(s, a, g, cfg)
    assert d["actuators_enabled"].all()
    assert n.energy.item() == 0.0
    assert not n.alive.any()
    assert n.velocity[0, 0] > 0



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_numerical_agreement():
    torch.manual_seed(42)
    cfg = PhysicsConfig()
    s = make_state((4, 256))
    s.velocity.uniform_(-1, 1); s.angle.uniform_(-3, 3)
    s.angular_velocity.uniform_(-1, 1); s.energy.uniform_(0, 100)
    a = torch.rand(4, 256, 2) * 2 - 1
    g = torch.randn(4, 2)
    cpu_next, _ = step(s, a, g, cfg)
    gpu_next, _ = step(s.to("cuda"), a.cuda(), g.cuda(), cfg)
    for f in ("position", "velocity", "angle", "angular_velocity", "energy"):
        torch.testing.assert_close(getattr(cpu_next, f), getattr(gpu_next, f), rtol=1e-5, atol=1e-6)
    assert torch.equal(cpu_next.alive, gpu_next.alive.cpu())
