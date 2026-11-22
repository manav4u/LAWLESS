"""Validated configuration definitions for the LAWLESS P1 environment."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _require_number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a numeric value, not bool")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _require_positive(name: str, value: Any) -> None:
    if _require_number(name, value) <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(name: str, value: Any) -> None:
    if _require_number(name, value) < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, not bool")


def _require_nonnegative_integer(name: str, value: Any) -> None:
    _require_integer(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_section(name: str, value: Any, expected_type: type) -> None:
    if not isinstance(value, expected_type):
        raise ValueError(
            f"{name} must be an instance of {expected_type.__name__}"
        )


def _require_sequence(name: str, value: Any) -> None:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be a list or tuple")


@dataclass
class WorldConfig:
    """Continuous-world geometry configuration."""

    width_m: float = 100.0
    height_m: float = 100.0
    obstacle_count: int = 20
    obstacle_radius_min_m: float = 1.0
    obstacle_radius_max_m: float = 3.0

    def validate(self) -> None:
        _require_positive("world.width_m", self.width_m)
        _require_positive("world.height_m", self.height_m)
        _require_nonnegative_integer(
            "world.obstacle_count", self.obstacle_count
        )
        _require_positive(
            "world.obstacle_radius_min_m",
            self.obstacle_radius_min_m,
        )
        _require_positive(
            "world.obstacle_radius_max_m",
            self.obstacle_radius_max_m,
        )

        if self.obstacle_radius_max_m < self.obstacle_radius_min_m:
            raise ValueError(
                "world.obstacle_radius_max_m must be >= "
                "world.obstacle_radius_min_m"
            )


@dataclass
class EmbodimentConfig:
    """Agent embodiment configuration."""

    agent_radius_m: float = 0.5

    def validate(self) -> None:
        _require_positive(
            "embodiment.agent_radius_m",
            self.agent_radius_m,
        )


@dataclass
class ObjectiveConfig:
    """Energy-core objective configuration."""

    core_count: int = 8
    core_radius_m: float = 0.75
    core_energy_j: float = 25.0
    extraction_radius_m: float = 3.0

    def validate(self) -> None:
        _require_nonnegative_integer(
            "objectives.core_count",
            self.core_count,
        )
        _require_positive(
            "objectives.core_radius_m",
            self.core_radius_m,
        )
        _require_positive(
            "objectives.core_energy_j",
            self.core_energy_j,
        )
        _require_positive(
            "objectives.extraction_radius_m",
            self.extraction_radius_m,
        )


@dataclass
class SensorConfig:
    """Eight-ray sensor configuration for the P1 contract."""

    lidar_ray_count: int = 8
    lidar_range_m: float = 20.0

    def validate(self) -> None:
        _require_integer(
            "sensors.lidar_ray_count",
            self.lidar_ray_count,
        )

        if self.lidar_ray_count != 8:
            raise ValueError(
                "sensors.lidar_ray_count must equal 8 "
                "for the locked P1 observation contract"
            )

        _require_positive(
            "sensors.lidar_range_m",
            self.lidar_range_m,
        )


@dataclass
class EpisodeConfig:
    """Episode timing and nonstationarity configuration."""

    episode_duration_s: float = 40.0
    shift_time_s: float = 20.0
    dt_s: float = 1.0 / 60.0
    gravity_m_s2: float = 4.0

    gravity_directions: tuple[tuple[float, float], ...] = (
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (-1.0, 0.0),
    )

    linear_damping_s_inv: tuple[float, ...] = (
        0.05,
        0.15,
        0.30,
    )

    angular_damping_s_inv: float = 0.20

    def validate(self) -> None:
        _require_positive(
            "episode.episode_duration_s",
            self.episode_duration_s,
        )
        _require_positive(
            "episode.dt_s",
            self.dt_s,
        )
        _require_nonnegative(
            "episode.shift_time_s",
            self.shift_time_s,
        )
        _require_positive(
            "episode.gravity_m_s2",
            self.gravity_m_s2,
        )
        _require_nonnegative(
            "episode.angular_damping_s_inv",
            self.angular_damping_s_inv,
        )

        if self.shift_time_s >= self.episode_duration_s:
            raise ValueError(
                "episode.shift_time_s must be less than "
                "episode.episode_duration_s"
            )

        _require_sequence(
            "episode.gravity_directions",
            self.gravity_directions,
        )

        if len(self.gravity_directions) == 0:
            raise ValueError(
                "episode.gravity_directions cannot be empty"
            )

        for i, direction in enumerate(self.gravity_directions):
            name = f"episode.gravity_directions[{i}]"
            _require_sequence(name, direction)

            if len(direction) != 2:
                raise ValueError(
                    f"{name} must contain exactly two components"
                )

            x = _require_number(f"{name}[0]", direction[0])
            y = _require_number(f"{name}[1]", direction[1])

            norm = math.hypot(x, y)

            if not math.isclose(
                norm, 1.0, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(f"{name} must be a unit vector")

        _require_sequence(
            "episode.linear_damping_s_inv",
            self.linear_damping_s_inv,
        )

        if len(self.linear_damping_s_inv) == 0:
            raise ValueError(
                "episode.linear_damping_s_inv cannot be empty"
            )

        for i, damping in enumerate(self.linear_damping_s_inv):
            _require_nonnegative(
                f"episode.linear_damping_s_inv[{i}]",
                damping,
            )

        shift_steps = self.shift_time_s / self.dt_s
        if not math.isfinite(shift_steps):
            raise ValueError("Shift timestep ratio must be finite")

        if not math.isclose(
            shift_steps,
            round(shift_steps),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "episode.shift_time_s must align with "
                "an integer timestep"
            )

        episode_steps = self.episode_duration_s / self.dt_s
        if not math.isfinite(episode_steps):
            raise ValueError("Episode timestep ratio must be finite")

        if not math.isclose(
            episode_steps,
            round(episode_steps),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "episode.episode_duration_s must align "
                "with an integer timestep"
            )


@dataclass
class EnvironmentConfig:
    """Top-level LAWLESS environment configuration."""

    seed: int = 42

    world: WorldConfig = field(default_factory=WorldConfig)
    embodiment: EmbodimentConfig = field(
        default_factory=EmbodimentConfig
    )
    objectives: ObjectiveConfig = field(
        default_factory=ObjectiveConfig
    )
    sensors: SensorConfig = field(default_factory=SensorConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)

    def validate(self) -> None:
        _require_nonnegative_integer("seed", self.seed)

        _require_section("world", self.world, WorldConfig)
        _require_section(
            "embodiment", self.embodiment, EmbodimentConfig
        )
        _require_section(
            "objectives", self.objectives, ObjectiveConfig
        )
        _require_section("sensors", self.sensors, SensorConfig)
        _require_section("episode", self.episode, EpisodeConfig)

        self.world.validate()
        self.embodiment.validate()
        self.objectives.validate()
        self.sensors.validate()
        self.episode.validate()


def _strict_dataclass(cls: type, data: dict[str, Any]):
    """Construct a dataclass while rejecting unknown keys."""

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping for {cls.__name__}")

    allowed = set(cls.__dataclass_fields__.keys())
    unknown = set(data.keys()) - allowed

    if unknown:
        raise ValueError(
            f"Unknown keys in {cls.__name__}: {sorted(unknown)}"
        )

    return cls(**data)


def config_from_dict(data: dict[str, Any]) -> EnvironmentConfig:
    """Build and validate EnvironmentConfig from a mapping."""

    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a mapping")

    allowed = set(EnvironmentConfig.__dataclass_fields__.keys())
    unknown = set(data.keys()) - allowed

    if unknown:
        raise ValueError(
            f"Unknown EnvironmentConfig keys: {sorted(unknown)}"
        )

    config = EnvironmentConfig(
        seed=data.get("seed", 42),
        world=_strict_dataclass(WorldConfig, data.get("world", {})),
        embodiment=_strict_dataclass(
            EmbodimentConfig, data.get("embodiment", {})
        ),
        objectives=_strict_dataclass(
            ObjectiveConfig, data.get("objectives", {})
        ),
        sensors=_strict_dataclass(
            SensorConfig, data.get("sensors", {})
        ),
        episode=_strict_dataclass(
            EpisodeConfig, data.get("episode", {})
        ),
    )

    # Validate before normalization so malformed sequence contents
    # cannot be silently converted or hidden.
    config.validate()

    # Normalize public sequence representation after validation.
    config.episode.gravity_directions = tuple(
        tuple(direction)
        for direction in config.episode.gravity_directions
    )
    config.episode.linear_damping_s_inv = tuple(
        config.episode.linear_damping_s_inv
    )

    return config


def load_config(path: str | Path) -> EnvironmentConfig:
    """Load and validate an environment configuration from YAML."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(
            "YAML configuration root must be a mapping"
        )

    return config_from_dict(data)
