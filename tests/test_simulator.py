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
