"""Expanded validation tests for LAWLESS P1.0."""

from pathlib import Path

import pytest

from lawless.world.config import (
    EnvironmentConfig,
    EpisodeConfig,
    EmbodimentConfig,
    ObjectiveConfig,
    SensorConfig,
    WorldConfig,
    config_from_dict,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "environment.yaml"


def test_load_default_yaml_config():
    config = load_config(CONFIG_PATH)

    assert isinstance(config, EnvironmentConfig)
    assert config.seed == 42

    assert config.world.width_m == 100.0
    assert config.world.height_m == 100.0
    assert config.world.obstacle_count == 20
    assert config.world.obstacle_radius_min_m == 1.0
    assert config.world.obstacle_radius_max_m == 3.0

    assert config.embodiment.agent_radius_m == 0.5

    assert config.objectives.core_count == 8
    assert config.objectives.core_radius_m == 0.75
    assert config.objectives.core_energy_j == 25.0
    assert config.objectives.extraction_radius_m == 3.0

    assert config.sensors.lidar_ray_count == 8
    assert config.sensors.lidar_range_m == 20.0

    assert config.episode.episode_duration_s == 40.0
    assert config.episode.shift_time_s == 20.0
    assert config.episode.dt_s == pytest.approx(1 / 60)
    assert config.episode.gravity_m_s2 == 4.0

    assert config.episode.gravity_directions == (
        (0.0, 1.0),
        (1.0, 0.0),
        (0.0, -1.0),
        (-1.0, 0.0),
    )

    assert config.episode.linear_damping_s_inv == (
        0.05,
        0.15,
        0.30,
    )

    assert config.episode.angular_damping_s_inv == 0.20


def test_shift_time_is_aligned_to_step():
    config = load_config(CONFIG_PATH)

    shift_steps = (
        config.episode.shift_time_s
        / config.episode.dt_s
    )

    assert shift_steps == pytest.approx(1200.0)


def test_zero_obstacles_and_cores_are_valid():
    config = EnvironmentConfig()

    config.world.obstacle_count = 0
    config.objectives.core_count = 0

    config.validate()


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
@pytest.mark.parametrize(
    "field_name",
    [
        "width_m",
        "height_m",
        "obstacle_radius_min_m",
        "obstacle_radius_max_m",
    ],
)
def test_world_nonfinite_numeric_values_rejected(
    field_name,
    value,
):
    config = EnvironmentConfig()
    setattr(config.world, field_name, value)

    with pytest.raises(ValueError, match="finite"):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_embodiment_nonfinite_radius_rejected(value):
    config = EnvironmentConfig()
    config.embodiment.agent_radius_m = value

    with pytest.raises(ValueError, match="finite"):
        config.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "core_radius_m",
        "core_energy_j",
        "extraction_radius_m",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_objective_nonfinite_numeric_values_rejected(
    field_name,
    value,
):
    config = EnvironmentConfig()
    setattr(config.objectives, field_name, value)

    with pytest.raises(ValueError, match="finite"):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_lidar_range_nonfinite_values_rejected(value):
    config = EnvironmentConfig()
    config.sensors.lidar_range_m = value

    with pytest.raises(ValueError, match="finite"):
        config.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "episode_duration_s",
        "shift_time_s",
        "dt_s",
        "gravity_m_s2",
        "angular_damping_s_inv",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_episode_nonfinite_numeric_values_rejected(
    field_name,
    value,
):
    config = EnvironmentConfig()
    setattr(config.episode, field_name, value)

    with pytest.raises(ValueError, match="finite"):
        config.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "width_m",
        "height_m",
        "obstacle_radius_min_m",
        "obstacle_radius_max_m",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ],
)
def test_world_bool_numeric_fields_rejected(
    field_name,
    value,
):
    config = EnvironmentConfig()
    setattr(config.world, field_name, value)

    with pytest.raises(ValueError, match="numeric"):
        config.validate()


@pytest.mark.parametrize(
    "field_name",
    [
        "obstacle_count",
    ],
)
@pytest.mark.parametrize(
    "value",
    [
        1.5,
        True,
        False,
    ],
)
def test_world_integer_fields_reject_nonintegers(
    field_name,
    value,
):
    config = EnvironmentConfig()
    setattr(config.world, field_name, value)

    with pytest.raises(ValueError, match="integer"):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [
        1.5,
        True,
        False,
    ],
)
def test_core_count_rejects_nonintegers(value):
    config = EnvironmentConfig()
    config.objectives.core_count = value

    with pytest.raises(ValueError, match="integer"):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        42.0,
        1.5,
    ],
)
def test_seed_rejects_bool_and_noninteger(value):
    config = EnvironmentConfig()
    config.seed = value

    with pytest.raises(ValueError, match="integer"):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [
        4,
        7,
        9,
        0,
        -8,
        8.0,
        True,
    ],
)
def test_lidar_ray_count_must_be_exactly_eight(value):
    config = EnvironmentConfig()
    config.sensors.lidar_ray_count = value

    with pytest.raises(ValueError):
        config.validate()


def test_default_lidar_count_is_eight():
    config = EnvironmentConfig()

    assert config.sensors.lidar_ray_count == 8

    config.validate()


@pytest.mark.parametrize(
    "section",
    [
        "world",
        "embodiment",
        "objectives",
        "sensors",
        "episode",
    ],
)
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        "invalid",
        123,
    ],
)
def test_malformed_section_instances_rejected(
    section,
    invalid,
):
    config = EnvironmentConfig()
    setattr(config, section, invalid)

    with pytest.raises(ValueError, match=section):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not-a-sequence",
        123,
        {},
    ],
)
def test_gravity_direction_container_rejected(value):
    config = EnvironmentConfig()
    config.episode.gravity_directions = value

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not-a-sequence",
        123,
        {},
    ],
)
def test_linear_damping_container_rejected(value):
    config = EnvironmentConfig()
    config.episode.linear_damping_s_inv = value

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "direction",
    [
        (1.0,),
        (1.0, 0.0, 0.0),
        (float("nan"), 0.0),
        (float("inf"), 0.0),
        (True, 0.0),
        "xy",
    ],
)
def test_invalid_gravity_direction_entries_rejected(direction):
    config = EnvironmentConfig()
    config.episode.gravity_directions = (direction,)

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [
        1.5,
        True,
        False,
    ],
)
def test_lidar_ray_count_type_rejected(value):
    config = EnvironmentConfig()
    config.sensors.lidar_ray_count = value

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [
        -1,
        1.5,
        True,
    ],
)
def test_seed_type_rejected(value):
    config = EnvironmentConfig()
    config.seed = value

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_world_width_is_rejected():
    config = EnvironmentConfig()
    config.world.width_m = 0.0

    with pytest.raises(ValueError):
        config.validate()


def test_invalid_gravity_direction_is_rejected():
    config = EnvironmentConfig()
    config.episode.gravity_directions = ((0.0, 0.0),)

    with pytest.raises(ValueError):
        config.validate()


def test_shift_time_must_align_to_timestep():
    config = EnvironmentConfig()
    config.episode.shift_time_s = 20.01

    with pytest.raises(ValueError):
        config.validate()


def test_unknown_yaml_configuration_keys_are_rejected(
    tmp_path,
):
    invalid_config = tmp_path / "invalid.yaml"

    invalid_config.write_text(
        """
seed: 42
world:
  width_m: 100.0
  typo_width: 500.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown keys"):
        load_config(invalid_config)


@pytest.mark.parametrize(
    "section",
    [
        "world",
        "embodiment",
        "objectives",
        "sensors",
        "episode",
    ],
)
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        "bad",
        123,
    ],
)
def test_config_mapping_rejects_malformed_sections(
    section,
    invalid,
):
    with pytest.raises(ValueError):
        config_from_dict({section: invalid})


def test_config_mapping_rejects_nonmapping_root():
    with pytest.raises(ValueError, match="root"):
        config_from_dict([])


def test_yaml_rejects_nonmapping_root(tmp_path):
    path = tmp_path / "bad_root.yaml"
    path.write_text("- item\n- item2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root"):
        load_config(path)


def test_yaml_rejects_malformed_section(tmp_path):
    path = tmp_path / "bad_section.yaml"
    path.write_text(
        """
world: null
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
    ],
)
def test_linear_damping_invalid_values_rejected(value):
    config = EnvironmentConfig()
    config.episode.linear_damping_s_inv = (value,)

    with pytest.raises(ValueError):
        config.validate()


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
    ],
)
def test_gravity_magnitude_invalid_values_rejected(value):
    config = EnvironmentConfig()
    config.episode.gravity_m_s2 = value

    with pytest.raises(ValueError):
        config.validate()


def test_negative_obstacle_count_rejected():
    config = EnvironmentConfig()
    config.world.obstacle_count = -1

    with pytest.raises(ValueError):
        config.validate()


def test_negative_core_count_rejected():
    config = EnvironmentConfig()
    config.objectives.core_count = -1

    with pytest.raises(ValueError):
        config.validate()
