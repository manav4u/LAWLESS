import pytest
from lawless.physics.constants import PhysicsConfig


def test_config_rejects_invalid_timestep():
    with pytest.raises(ValueError):
        PhysicsConfig(dt=0)


def test_config_rejects_initial_energy_over_max():
    with pytest.raises(ValueError):
        PhysicsConfig(initial_energy=101, max_energy=100)
