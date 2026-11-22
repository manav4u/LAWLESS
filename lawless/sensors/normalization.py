
"""Fixed, environment-independent scales for LAWLESS observations.

These constants are part of the P1.6 observation interface. They must not
be derived from an episode's PhysicsConfig, WorldConfig, or SensorConfig.
"""

OBSERVATION_SIZE = 23

# Target geometry.
RELATIVE_POSITION_SCALE_M = 50.0
TARGET_DISTANCE_SCALE_M = 100.0

# Motion.
LINEAR_VELOCITY_SCALE_MPS = 12.0
ANGULAR_VELOCITY_SCALE_RADPS = 4.0
LINEAR_ACCELERATION_SCALE_MPS2 = 16.0

# Energy. This remains fixed even if an environment uses a different
# PhysicsConfig.max_energy.
ENERGY_SCALE_J = 100.0

# Observation channel layout.
TARGET_RELATIVE_POSITION = slice(0, 2)
TARGET_DISTANCE = 2
TARGET_BEARING_SIN = 3
TARGET_BEARING_COS = 4
BODY_LINEAR_VELOCITY = slice(5, 7)
ANGULAR_VELOCITY = 7
HEADING_SIN = 8
HEADING_COS = 9
LIDAR = slice(10, 18)
PREVIOUS_ACTIONS = slice(18, 20)
BODY_LINEAR_ACCELERATION = slice(20, 22)
REMAINING_ENERGY = 22
