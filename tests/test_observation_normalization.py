
"""Tests for the fixed P1.6 observation normalization contract."""

import pytest

from lawless.sensors.normalization import (
    ANGULAR_VELOCITY,
    ANGULAR_VELOCITY_SCALE_RADPS,
    BODY_LINEAR_ACCELERATION,
    BODY_LINEAR_VELOCITY,
    ENERGY_SCALE_J,
    HEADING_COS,
    HEADING_SIN,
    LIDAR,
    LINEAR_ACCELERATION_SCALE_MPS2,
    LINEAR_VELOCITY_SCALE_MPS,
    OBSERVATION_SIZE,
    PREVIOUS_ACTIONS,
    RELATIVE_POSITION_SCALE_M,
    REMAINING_ENERGY,
    TARGET_BEARING_COS,
    TARGET_BEARING_SIN,
    TARGET_DISTANCE,
    TARGET_DISTANCE_SCALE_M,
    TARGET_RELATIVE_POSITION,
)


def test_observation_size_is_exactly_23():
    assert OBSERVATION_SIZE == 23


def test_target_geometry_channel_layout():
    assert TARGET_RELATIVE_POSITION == slice(0, 2)
    assert TARGET_DISTANCE == 2
    assert TARGET_BEARING_SIN == 3
    assert TARGET_BEARING_COS == 4


def test_motion_channel_layout():
    assert BODY_LINEAR_VELOCITY == slice(5, 7)
    assert ANGULAR_VELOCITY == 7
    assert HEADING_SIN == 8
    assert HEADING_COS == 9


def test_sensor_and_action_channel_layout():
    assert LIDAR == slice(10, 18)
    assert PREVIOUS_ACTIONS == slice(18, 20)
    assert BODY_LINEAR_ACCELERATION == slice(20, 22)
    assert REMAINING_ENERGY == 22


@pytest.mark.parametrize(
    ("name", "actual", "expected"),
    [
        (
            "relative position",
            RELATIVE_POSITION_SCALE_M,
            50.0,
        ),
        (
            "target distance",
            TARGET_DISTANCE_SCALE_M,
            100.0,
        ),
        (
            "linear velocity",
            LINEAR_VELOCITY_SCALE_MPS,
            12.0,
        ),
        (
            "angular velocity",
            ANGULAR_VELOCITY_SCALE_RADPS,
            4.0,
        ),
        (
            "linear acceleration",
            LINEAR_ACCELERATION_SCALE_MPS2,
            16.0,
        ),
        (
            "energy",
            ENERGY_SCALE_J,
            100.0,
        ),
    ],
)
def test_fixed_normalization_scales(name, actual, expected):
    assert actual == pytest.approx(expected), name


def test_channel_slices_are_contiguous_and_cover_all_channels():
    channel_indices = (
        list(range(TARGET_RELATIVE_POSITION.start,
                   TARGET_RELATIVE_POSITION.stop))
        + [TARGET_DISTANCE, TARGET_BEARING_SIN, TARGET_BEARING_COS]
        + list(range(BODY_LINEAR_VELOCITY.start,
                     BODY_LINEAR_VELOCITY.stop))
        + [ANGULAR_VELOCITY, HEADING_SIN, HEADING_COS]
        + list(range(LIDAR.start, LIDAR.stop))
        + list(range(PREVIOUS_ACTIONS.start, PREVIOUS_ACTIONS.stop))
        + list(range(BODY_LINEAR_ACCELERATION.start,
                     BODY_LINEAR_ACCELERATION.stop))
        + [REMAINING_ENERGY]
    )

    assert channel_indices == list(range(OBSERVATION_SIZE))


def test_energy_scale_is_fixed_independently_of_environment_configuration():
    # The public normalization contract is a fixed 100 J denominator.
    # Environment-specific PhysicsConfig values must not alter this constant.
    assert ENERGY_SCALE_J == 100.0
