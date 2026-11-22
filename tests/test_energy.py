import torch

from lawless.physics.constants import PhysicsConfig
from lawless.physics.energy import actuator_energy_step


def test_energy_accounting():
    cfg = PhysicsConfig(dt=0.1, thrust_power=1.5, torque_power=0.5)
    energy = torch.tensor([[100.0]], dtype=torch.float64)
    throttle = torch.tensor([[1.0]], dtype=torch.float64)
    torque = torch.tensor([[-1.0]], dtype=torch.float64)
    next_energy, enabled = actuator_energy_step(energy, throttle, torque, cfg)
    torch.testing.assert_close(next_energy, torch.tensor([[99.8]], dtype=torch.float64))
    assert enabled.item()


def test_energy_never_negative_and_zero_disables():
    cfg = PhysicsConfig(dt=1.0, thrust_power=100.0, torque_power=100.0)
    energy = torch.tensor([[0.1], [0.0]], dtype=torch.float64)
    command = torch.ones_like(energy)
    next_energy, enabled = actuator_energy_step(energy, command, command, cfg)
    assert torch.all(next_energy >= 0)
    assert enabled[:, 0].tolist() == [True, False]
    assert next_energy[0, 0].item() == 0.0
