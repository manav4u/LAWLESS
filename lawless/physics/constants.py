"""Physical constants and validated configuration."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsConfig:
    dt: float = 1.0 / 60.0
    mass: float = 1.0
    inertia: float = 0.5
    max_thrust: float = 12.0
    max_torque: float = 2.0
    linear_damping: float = 0.15
    angular_damping: float = 0.20
    initial_energy: float = 100.0
    max_energy: float = 100.0
    thrust_power: float = 1.5
    torque_power: float = 0.5

    def __post_init__(self) -> None:
        positive = {
            "dt": self.dt,
            "mass": self.mass,
            "inertia": self.inertia,
            "max_thrust": self.max_thrust,
            "max_torque": self.max_torque,
            "max_energy": self.max_energy,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        nonnegative = {
            "linear_damping": self.linear_damping,
            "angular_damping": self.angular_damping,
            "initial_energy": self.initial_energy,
            "thrust_power": self.thrust_power,
            "torque_power": self.torque_power,
        }
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.initial_energy > self.max_energy:
            raise ValueError("initial_energy cannot exceed max_energy")
