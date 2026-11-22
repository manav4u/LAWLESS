
"""P1.6 observation assembly for LAWLESS episodes.

Observations have shape (*batch_shape, 23). The extraction center is the
target throughout the episode. Energy cores remain separate collection
objectives and never redirect the observation target.

Lidar input is expected to be the normalized output of sense_lidar().
Alive rows preserve supplied lidar values exactly. Dead rows are
canonicalized to eight ones.
"""

from __future__ import annotations

import math

import torch

from lawless.sensors.normalization import (
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
    REMAINING_ENERGY,
    RELATIVE_POSITION_SCALE_M,
    TARGET_BEARING_COS,
    TARGET_BEARING_SIN,
    TARGET_DISTANCE,
    TARGET_DISTANCE_SCALE_M,
    TARGET_RELATIVE_POSITION,
)


_FLOAT_DTYPES = (torch.float32, torch.float64)
_LOG2_POSITION_SCALE = math.log2(RELATIVE_POSITION_SCALE_M)
_LOG2_DISTANCE_SCALE = math.log2(TARGET_DISTANCE_SCALE_M)


def _require_float_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {tuple(value.shape)}"
        )
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if value.device != device:
        raise ValueError(
            f"{name} must be on device {device}, got {value.device}"
        )
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _require_bool_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, ...],
    *,
    device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {tuple(value.shape)}"
        )
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool")
    if value.device != device:
        raise ValueError(
            f"{name} must be on device {device}, got {value.device}"
        )


def _scaled_axis_difference(
    target: torch.Tensor,
    origin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a safe difference representation (delta_scaled, exponent).

    target and origin have the same shape. Scaling is by an exact power of
    two chosen independently for each coordinate axis. This preserves
    adjacent representable values while preventing subtraction overflow
    for opposite-sign, near-maximum finite operands.

    The represented difference is:
        (target - origin) == delta_scaled * 2**exponent

    The equality is mathematical; the represented delta is computed without
    constructing the potentially overflowing unscaled subtraction.
    """
    largest = torch.maximum(target.abs(), origin.abs())
    _, exponent = torch.frexp(largest)

    target_scaled = torch.ldexp(target, -exponent)
    origin_scaled = torch.ldexp(origin, -exponent)
    delta_scaled = target_scaled - origin_scaled
    return delta_scaled, exponent


def _common_scaled_displacement(
    target: torch.Tensor,
    origin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return dx, dy, and a common binary exponent for the displacement.

    Each axis is differenced independently first. This is important when,
    for example, both X coordinates are extremely large and equal while
    Y contains a very small but nonzero displacement.

    The returned components represent the true displacement multiplied by
    2**(-common_exponent). At least one nonzero component is normalized to
    a moderate magnitude. A zero displacement is represented by (0, 0).
    """
    dx_scaled, exp_x = _scaled_axis_difference(
        target[..., 0], origin[..., 0]
    )
    dy_scaled, exp_y = _scaled_axis_difference(
        target[..., 1], origin[..., 1]
    )

    _, delta_exp_x = torch.frexp(dx_scaled.abs())
    _, delta_exp_y = torch.frexp(dy_scaled.abs())

    # A zero component must not influence the common exponent.
    lowest = torch.full_like(exp_x, torch.iinfo(exp_x.dtype).min // 2)
    effective_x = torch.where(
        dx_scaled != 0, exp_x + delta_exp_x, lowest
    )
    effective_y = torch.where(
        dy_scaled != 0, exp_y + delta_exp_y, lowest
    )

    common_exp = torch.maximum(effective_x, effective_y)
    any_nonzero = (dx_scaled != 0) | (dy_scaled != 0)
    common_exp = torch.where(any_nonzero, common_exp, torch.zeros_like(common_exp))

    dx = torch.ldexp(dx_scaled, exp_x - common_exp)
    dy = torch.ldexp(dy_scaled, exp_y - common_exp)
    return dx, dy, common_exp


def _scaled_signed_clip(
    value: torch.Tensor,
    exponent: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Compute clip(value * 2**exponent / scale, -1, 1) safely.

    The saturation decision is made in log2 space. ldexp is only applied
    with the original exponent for entries known not to saturate, avoiding
    overflow when the represented physical value is extremely large.
    """
    absolute = value.abs()
    nonzero = absolute > 0

    log_magnitude = (
        torch.log2(torch.where(nonzero, absolute, torch.ones_like(absolute)))
        + exponent.to(dtype=absolute.dtype)
        - math.log2(scale)
    )
    saturated = nonzero & (log_magnitude >= 0)

    safe_exponent = torch.where(
        saturated, torch.zeros_like(exponent), exponent
    )
    unsaturated = torch.ldexp(value / scale, safe_exponent)

    return torch.where(
        saturated,
        torch.sign(value),
        unsaturated,
    )


def _scaled_nonnegative_clip(
    magnitude: torch.Tensor,
    exponent: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Compute min(magnitude * 2**exponent / scale, 1) safely."""
    nonzero = magnitude > 0
    log_magnitude = (
        torch.log2(
            torch.where(nonzero, magnitude, torch.ones_like(magnitude))
        )
        + exponent.to(dtype=magnitude.dtype)
        - math.log2(scale)
    )
    saturated = nonzero & (log_magnitude >= 0)

    safe_exponent = torch.where(
        saturated, torch.zeros_like(exponent), exponent
    )
    unsaturated = torch.ldexp(magnitude / scale, safe_exponent)
    return torch.where(saturated, torch.ones_like(unsaturated), unsaturated)


def _rotate_body(
    x_world: torch.Tensor,
    y_world: torch.Tensor,
    angle: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate world-frame vectors into the body frame.

    +X is forward, +Y is left, and positive heading is counter-clockwise.
    Inputs are expected to be scaled/bounded before calling this helper.
    """
    cosine = torch.cos(angle)
    sine = torch.sin(angle)
    x_body = x_world * cosine + y_world * sine
    y_body = -x_world * sine + y_world * cosine
    return x_body, y_body


def _normalize_rotated_vector(
    x_world: torch.Tensor,
    y_world: torch.Tensor,
    angle: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate a finite vector and normalize it without large products.

    Dividing by the largest component first bounds the rotation. Each
    component is then either clipped by a safe threshold comparison or
    multiplied only in the unsaturated range.
    """
    magnitude = torch.maximum(x_world.abs(), y_world.abs())
    nonzero = magnitude > 0
    safe_magnitude = torch.where(
        nonzero, magnitude, torch.ones_like(magnitude)
    )

    x_scaled = x_world / safe_magnitude
    y_scaled = y_world / safe_magnitude
    x_rotated, y_rotated = _rotate_body(x_scaled, y_scaled, angle)

    def normalize_component(component: torch.Tensor) -> torch.Tensor:
        threshold = scale / safe_magnitude
        saturated = nonzero & (component.abs() >= threshold)

        # For saturated entries, zero the magnitude before multiplication.
        # The remaining products are mathematically bounded by one.
        product_magnitude = torch.where(
            saturated, torch.zeros_like(magnitude), magnitude
        )
        unsaturated = component * (product_magnitude / scale)

        return torch.where(
            saturated,
            torch.sign(component),
            unsaturated,
        )

    return normalize_component(x_rotated), normalize_component(y_rotated)


def assemble_observations(
    episode_state,
    lidar_distances: torch.Tensor,
) -> torch.Tensor:
    """Assemble the 23-channel P1.6 observation tensor.

    Args:
        episode_state:
            An initialized AgentEpisodeState with positive, nonempty
            batch dimensions.
        lidar_distances:
            Already-normalized lidar values with shape
            (*batch_shape, 8), dtype/device matching the physics state.

    Returns:
        A newly allocated tensor with shape (*batch_shape, 23), on the
        physics state's device and with its floating-point dtype.

    The extraction center is always the target. Core collection does not
    change the target. Alive lidar values are preserved exactly; dead rows
    are canonicalized to all ones.
    """
    from lawless.world.episode_state import AgentEpisodeState

    if not isinstance(episode_state, AgentEpisodeState):
        raise TypeError("episode_state must be an AgentEpisodeState")

    # Access the physical position tensor before batch_shape: the latter
    # derives its dimensions through the embodiment and physics state.
    embodiment = episode_state.embodiment
    physics = embodiment.physics

    if not isinstance(physics.position, torch.Tensor):
        raise TypeError("physics.position must be a torch.Tensor")

    dtype = physics.position.dtype
    device = physics.position.device

    if dtype not in _FLOAT_DTYPES:
        raise TypeError("physics state must use float32 or float64")

    batch_shape = tuple(episode_state.batch_shape)
    if not batch_shape or any(
        not isinstance(size, int) or size <= 0 for size in batch_shape
    ):
        raise ValueError(
            "episode_state must have positive, nonempty batch dimensions"
        )

    # Floating-point dtype was validated before deriving batch_shape.

    # Validate mutable tensor fields individually below. Do not call
    # AgentState.validate_shapes() first: malformed replacement fields
    # must produce the specific type/shape/dtype/device errors from the
    # observation boundary, not incidental attribute errors.

    vector_shape = batch_shape + (2,)
    scalar_shape = batch_shape + (1,)

    _require_float_tensor(
        "physics.position", physics.position, vector_shape,
        dtype=dtype, device=device
    )
    _require_float_tensor(
        "physics.velocity", physics.velocity, vector_shape,
        dtype=dtype, device=device
    )
    _require_float_tensor(
        "physics.angle", physics.angle, scalar_shape,
        dtype=dtype, device=device
    )
    _require_float_tensor(
        "physics.angular_velocity", physics.angular_velocity, scalar_shape,
        dtype=dtype, device=device
    )
    _require_float_tensor(
        "physics.energy", physics.energy, scalar_shape,
        dtype=dtype, device=device
    )
    _require_bool_tensor(
        "physics.alive", physics.alive, scalar_shape, device=device
    )

    _require_float_tensor(
        "embodiment.previous_actions", embodiment.previous_actions,
        vector_shape, dtype=dtype, device=device
    )
    _require_float_tensor(
        "embodiment.measured_acceleration",
        embodiment.measured_acceleration,
        vector_shape, dtype=dtype, device=device
    )
    _require_bool_tensor(
        "embodiment.extracted", embodiment.extracted,
        scalar_shape, device=device
    )
    _require_bool_tensor(
        "embodiment.failed", embodiment.failed,
        scalar_shape, device=device
    )

    if torch.any((embodiment.previous_actions < -1) |
                 (embodiment.previous_actions > 1)).item():
        raise ValueError("previous_actions must lie in [-1, 1]")

    if torch.any(embodiment.extracted & embodiment.failed).item():
        raise ValueError("extracted and failed masks must not overlap")
    if torch.any((embodiment.extracted | embodiment.failed) &
                 physics.alive).item():
        raise ValueError("terminal annotations require physics.alive=False")

    world = episode_state.generated_world
    extraction_center = world.extraction_center
    _require_float_tensor(
        "generated_world.extraction_center",
        extraction_center,
        (2,),
        dtype=dtype,
        device=device,
    )

    objectives = episode_state.objectives
    collected = objectives.collected
    if not isinstance(collected, torch.Tensor):
        raise TypeError("objectives.collected must be a torch.Tensor")
    if collected.dtype != torch.bool:
        raise TypeError("objectives.collected must have dtype torch.bool")
    if collected.device != device:
        raise ValueError("objectives.collected must match the physics device")
    if tuple(collected.shape[:-1]) != batch_shape:
        raise ValueError("objectives.collected batch shape is inconsistent")

    lidar_shape = batch_shape + (8,)
    _require_float_tensor(
        "lidar_distances",
        lidar_distances,
        lidar_shape,
        dtype=dtype,
        device=device,
    )
    if torch.any((lidar_distances < 0) | (lidar_distances > 1)).item():
        raise ValueError("lidar_distances must lie in [0, 1]")

    # All original inputs have now passed type, shape, dtype, device,
    # finiteness, and range validation. Use float64 working views for
    # geometry, normalization, and trigonometry. These conversions do not
    # mutate or replace any episode-state tensors.
    position_work = physics.position.to(dtype=torch.float64)
    extraction_center_work = extraction_center.to(dtype=torch.float64)
    velocity_work = physics.velocity.to(dtype=torch.float64)
    angle_work = physics.angle.to(dtype=torch.float64)
    angular_velocity_work = physics.angular_velocity.to(dtype=torch.float64)
    energy_work = physics.energy.to(dtype=torch.float64)
    acceleration_work = embodiment.measured_acceleration.to(
        dtype=torch.float64
    )
    actions_work = embodiment.previous_actions.to(dtype=torch.float64)
    lidar_work = lidar_distances.to(dtype=torch.float64)

    # Compute extraction displacement without ever subtracting two
    # potentially extreme unscaled coordinates.
    target = extraction_center_work.expand(vector_shape)
    dx, dy, displacement_exponent = _common_scaled_displacement(
        target, position_work
    )

    angle = angle_work.squeeze(-1)
    x_body_scaled, y_body_scaled = _rotate_body(dx, dy, angle)

    displacement_norm = torch.sqrt(dx.square() + dy.square())
    distance = _scaled_nonnegative_clip(
        displacement_norm,
        displacement_exponent,
        TARGET_DISTANCE_SCALE_M,
    )

    body_norm = torch.sqrt(
        x_body_scaled.square() + y_body_scaled.square()
    )
    nonzero_displacement = body_norm > 0

    safe_body_norm = torch.where(
        nonzero_displacement,
        body_norm,
        torch.ones_like(body_norm),
    )
    bearing_sin = y_body_scaled / safe_body_norm
    bearing_cos = x_body_scaled / safe_body_norm
    bearing_sin = torch.where(
        nonzero_displacement, bearing_sin, torch.zeros_like(bearing_sin)
    )
    bearing_cos = torch.where(
        nonzero_displacement, bearing_cos, torch.ones_like(bearing_cos)
    )

    relative_x = _scaled_signed_clip(
        x_body_scaled,
        displacement_exponent,
        RELATIVE_POSITION_SCALE_M,
    )
    relative_y = _scaled_signed_clip(
        y_body_scaled,
        displacement_exponent,
        RELATIVE_POSITION_SCALE_M,
    )

    velocity_x, velocity_y = _normalize_rotated_vector(
        velocity_work[..., 0],
        velocity_work[..., 1],
        angle,
        LINEAR_VELOCITY_SCALE_MPS,
    )
    acceleration_x, acceleration_y = _normalize_rotated_vector(
        acceleration_work[..., 0],
        acceleration_work[..., 1],
        angle,
        LINEAR_ACCELERATION_SCALE_MPS2,
    )

    angular_velocity = (
        angular_velocity_work / ANGULAR_VELOCITY_SCALE_RADPS
    ).clamp(-1, 1)

    heading_sin = torch.sin(angle)
    heading_cos = torch.cos(angle)

    energy = (energy_work / ENERGY_SCALE_J).clamp(0, 1)

    alive = physics.alive.squeeze(-1)
    lidar = torch.where(
        alive.unsqueeze(-1),
        lidar_work,
        torch.ones_like(lidar_work),
    )

    # Zero every non-lidar channel for inactive agents. This includes
    # exhausted agents whose terminal-cause annotations may both be false.
    def mask_alive(value: torch.Tensor) -> torch.Tensor:
        mask = alive
        while mask.ndim < value.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, value, torch.zeros_like(value))

    channels = [
        mask_alive(relative_x.unsqueeze(-1)),
        mask_alive(relative_y.unsqueeze(-1)),
        mask_alive(distance.unsqueeze(-1)),
        mask_alive(bearing_sin.unsqueeze(-1)),
        mask_alive(bearing_cos.unsqueeze(-1)),
        mask_alive(velocity_x.unsqueeze(-1)),
        mask_alive(velocity_y.unsqueeze(-1)),
        mask_alive(angular_velocity),
        mask_alive(heading_sin.unsqueeze(-1)),
        mask_alive(heading_cos.unsqueeze(-1)),
        lidar,
        mask_alive(actions_work[..., 0:1]),
        mask_alive(actions_work[..., 1:2]),
        mask_alive(acceleration_x.unsqueeze(-1)),
        mask_alive(acceleration_y.unsqueeze(-1)),
        mask_alive(energy),
    ]

    observations_work = torch.cat(channels, dim=-1)
    if tuple(observations_work.shape) != batch_shape + (OBSERVATION_SIZE,):
        raise RuntimeError("internal observation assembly shape error")

    if not torch.isfinite(observations_work).all().item():
        raise RuntimeError("observation assembly produced non-finite values")

    # Return a fresh tensor on the original device and in the original
    # physics dtype. Float32 lidar and action values are exactly representable
    # in float64, so this round-trip preserves their original values exactly.
    observations = observations_work.to(dtype=dtype)

    if not torch.isfinite(observations).all().item():
        raise RuntimeError("observation output contains non-finite values")

    return observations
