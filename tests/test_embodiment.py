import pytest
import torch

from lawless.physics.state import AgentState, make_state
from lawless.world.embodiment import AgentEmbodimentState


def _state(shape=(4,), *, device="cpu", dtype=torch.float32):
    return make_state(shape, device=device, dtype=dtype)


def _replace_alive(state, alive):
    return AgentState(
        position=state.position,
        velocity=state.velocity,
        angle=state.angle,
        angular_velocity=state.angular_velocity,
        energy=state.energy,
        alive=alive,
    )


@pytest.mark.parametrize("shape", [(5,), (2, 3), (1, 1)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initialize_supports_arbitrary_batch_shapes_and_dtypes(shape, dtype):
    physics = _state(shape, dtype=dtype)
    embodiment = AgentEmbodimentState.initialize(physics)

    assert embodiment.batch_shape == shape
    assert embodiment.previous_actions.shape == shape + (2,)
    assert embodiment.measured_acceleration.shape == shape + (2,)
    assert embodiment.extracted.shape == shape + (1,)
    assert embodiment.failed.shape == shape + (1,)

    assert embodiment.previous_actions.dtype == dtype
    assert embodiment.measured_acceleration.dtype == dtype
    assert not embodiment.extracted.any()
    assert not embodiment.failed.any()
    assert embodiment.physics.alive.all()


def test_initialize_does_not_relabel_preexisting_dead_agents():
    physics = _state((3,))
    alive = torch.tensor([[True], [False], [True]])
    embodiment = AgentEmbodimentState.initialize(_replace_alive(physics, alive))

    assert torch.equal(embodiment.physics.alive, alive)
    assert not embodiment.extracted.any()
    assert not embodiment.failed.any()


def test_constructor_takes_ownership_of_independent_tensor_storage():
    physics = _state((2,))
    actions = torch.tensor([[0.2, -0.3], [0.4, 0.5]])
    acceleration = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    extracted = torch.zeros((2, 1), dtype=torch.bool)
    failed = torch.zeros((2, 1), dtype=torch.bool)

    embodiment = AgentEmbodimentState(
        physics, actions, acceleration, extracted, failed
    )

    actions[0, 0] = 0.9
    acceleration[0, 0] = 99.0
    extracted[0, 0] = True
    physics.position[0, 0] = 123.0

    assert embodiment.previous_actions[0, 0].item() == pytest.approx(0.2)
    assert embodiment.measured_acceleration[0, 0].item() == pytest.approx(1.0)
    assert not embodiment.extracted[0, 0]
    assert embodiment.physics.position[0, 0].item() != pytest.approx(123.0)


def test_clone_has_independent_storage():
    original = AgentEmbodimentState.initialize(_state((2,)))
    cloned = original.clone()

    cloned.previous_actions[0, 0] = 0.5
    cloned.measured_acceleration[0, 0] = 3.0
    cloned.physics.position[0, 0] = 7.0

    assert original.previous_actions[0, 0].item() == 0.0
    assert original.measured_acceleration[0, 0].item() == 0.0
    assert original.physics.position[0, 0].item() != pytest.approx(7.0)


def test_measurements_are_explicit_and_do_not_use_p0_diagnostics():
    embodiment = AgentEmbodimentState.initialize(_state((2,)))

    actions = torch.tensor([[1.0, -1.0], [0.25, 0.5]])
    measured = torch.tensor([[2.0, 3.0], [-4.0, 5.0]])

    updated = embodiment.record_transition_measurements(
        applied_actions=actions,
        measured_acceleration=measured,
    )

    assert torch.equal(updated.previous_actions, actions)
    assert torch.equal(updated.measured_acceleration, measured)
    assert torch.count_nonzero(embodiment.measured_acceleration) == 0


@pytest.mark.parametrize(
    "actions, measured, error",
    [
        (torch.zeros((2, 3)), torch.zeros((2, 2)), ValueError),
        (torch.full((2, 2), 2.0), torch.zeros((2, 2)), ValueError),
        (torch.zeros((2, 2)), torch.full((2, 2), float("nan")), ValueError),
        (torch.zeros((2, 2), dtype=torch.float64),
         torch.zeros((2, 2), dtype=torch.float64), TypeError),
    ],
)
def test_measurement_contract_rejects_invalid_inputs(actions, measured, error):
    embodiment = AgentEmbodimentState.initialize(_state((2,)))
    with pytest.raises(error):
        embodiment.record_transition_measurements(
            applied_actions=actions,
            measured_acceleration=measured,
        )


def test_physical_replacement_preserves_measurements_and_masks():
    embodiment = AgentEmbodimentState.initialize(_state((2,)))
    measured = embodiment.record_transition_measurements(
        applied_actions=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        measured_acceleration=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )

    next_physics = measured.physics.clone()
    next_physics.position = next_physics.position + 1.0
    updated = measured.replace_physics(next_physics)

    assert torch.equal(updated.physics.position, next_physics.position)
    assert torch.equal(updated.previous_actions, measured.previous_actions)
    assert torch.equal(
        updated.measured_acceleration, measured.measured_acceleration
    )
    assert torch.equal(updated.extracted, measured.extracted)
    assert torch.equal(updated.failed, measured.failed)


def test_physical_replacement_cannot_resurrect_dead_agent():
    embodiment = AgentEmbodimentState.initialize(_state((2,)))
    dead_alive = torch.tensor([[False], [True]])
    dead_physics = _replace_alive(embodiment.physics, dead_alive)
    dead = embodiment.replace_physics(dead_physics)

    resurrected = _replace_alive(
        dead.physics, torch.tensor([[True], [True]])
    )
    with pytest.raises(ValueError, match="reactivate"):
        dead.replace_physics(resurrected)


def test_physical_replacement_must_keep_previously_dead_agents_frozen():
    embodiment = AgentEmbodimentState.initialize(_state((2,)))
    dead = embodiment.replace_physics(
        _replace_alive(
            embodiment.physics,
            torch.tensor([[False], [True]]),
        )
    )

    changed = dead.physics.clone()
    changed.position[0, 0] += 1.0

    with pytest.raises(ValueError, match="frozen physical state"):
        dead.replace_physics(changed)


def test_extraction_is_terminal_and_idempotent():
    embodiment = AgentEmbodimentState.initialize(_state((3,)))
    request = torch.tensor([[True], [False], [False]])

    extracted = embodiment.mark_extracted(request)
    repeated = extracted.mark_extracted(request)

    assert not extracted.physics.alive[0, 0]
    assert extracted.extracted[0, 0]
    assert not extracted.failed.any()
    assert torch.equal(repeated.extracted, extracted.extracted)
    assert torch.equal(repeated.physics.alive, extracted.physics.alive)


def test_failure_is_terminal_and_idempotent():
    embodiment = AgentEmbodimentState.initialize(_state((3,)))
    request = torch.tensor([[False], [True], [False]])

    failed = embodiment.mark_failed(request)
    repeated = failed.mark_failed(request)

    assert not failed.physics.alive[1, 0]
    assert failed.failed[1, 0]
    assert not failed.extracted.any()
    assert torch.equal(repeated.failed, failed.failed)


def test_conflicting_terminal_causes_are_rejected():
    embodiment = AgentEmbodimentState.initialize(_state((2,)))
    extracted = embodiment.mark_extracted(
        torch.tensor([[True], [False]])
    )

    with pytest.raises(ValueError, match="already extracted"):
        extracted.mark_failed(torch.tensor([[True], [False]]))

    with pytest.raises(ValueError, match="together"):
        embodiment.set_terminal(
            extracted=torch.tensor([[True], [False]]),
            failed=torch.tensor([[True], [False]]),
        )


def test_energy_exhausted_agent_is_not_relabelled_as_failed_or_extracted():
    embodiment = AgentEmbodimentState.initialize(_state((2,)))
    exhausted = embodiment.replace_physics(
        _replace_alive(
            embodiment.physics,
            torch.tensor([[False], [True]]),
        )
    )

    assert not exhausted.physics.alive[0, 0]
    assert not exhausted.failed[0, 0]
    assert not exhausted.extracted[0, 0]

    with pytest.raises(ValueError, match="inactive"):
        exhausted.mark_failed(torch.tensor([[True], [False]]))

    with pytest.raises(ValueError, match="inactive"):
        exhausted.mark_extracted(torch.tensor([[True], [False]]))


def test_terminal_agent_cannot_be_reactivated_by_physical_replacement():
    embodiment = AgentEmbodimentState.initialize(_state((1,)))
    extracted = embodiment.mark_extracted(torch.tensor([[True]]))

    with pytest.raises(ValueError, match="reactivate"):
        extracted.replace_physics(
            _replace_alive(extracted.physics, torch.tensor([[True]]))
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_dtype_device_and_clone_contract(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    physics = _state((2, 3), device="cuda", dtype=dtype)
    embodiment = AgentEmbodimentState.initialize(physics)

    assert embodiment.physics.position.device.type == "cuda"
    assert embodiment.previous_actions.device.type == "cuda"
    assert embodiment.measured_acceleration.dtype == dtype

    cloned = embodiment.clone()
    cloned.previous_actions[0, 0, 0] = 0.75
    assert embodiment.previous_actions[0, 0, 0].item() == 0.0


def test_replacement_rejects_batch_shape_device_or_dtype_mismatch():
    embodiment = AgentEmbodimentState.initialize(_state((2,)))

    with pytest.raises(ValueError, match="batch_shape"):
        embodiment.replace_physics(_state((3,)))

    with pytest.raises(ValueError, match="dtype"):
        embodiment.replace_physics(_state((2,), dtype=torch.float64))

    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="device"):
            embodiment.replace_physics(_state((2,), device="cuda"))


from lawless.physics.constants import PhysicsConfig
from lawless.physics.simulator import step


_CONTINUOUS_FIELDS = (
    "position",
    "velocity",
    "angle",
    "angular_velocity",
    "energy",
)


def _state_with_field_changed(state, field, value):
    values = {
        "position": state.position.clone(),
        "velocity": state.velocity.clone(),
        "angle": state.angle.clone(),
        "angular_velocity": state.angular_velocity.clone(),
        "energy": state.energy.clone(),
        "alive": state.alive.clone(),
    }
    values[field] = value
    return AgentState(**values)


@pytest.mark.parametrize("field", _CONTINUOUS_FIELDS)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_initialize_rejects_nonfinite_values_in_every_physics_field(
    field, bad_value
):
    state = _state((2,))
    value = getattr(state, field).clone()
    value.reshape(-1)[0] = bad_value
    invalid = _state_with_field_changed(state, field, value)

    with pytest.raises(ValueError, match="finite"):
        AgentEmbodimentState.initialize(invalid)


@pytest.mark.parametrize("field", _CONTINUOUS_FIELDS)
def test_initialize_rejects_non_float32_float64_physics_dtypes(field):
    state = _state((2,))
    value = getattr(state, field).to(torch.int64)
    invalid = _state_with_field_changed(state, field, value)

    with pytest.raises(TypeError, match="float32 or torch.float64"):
        AgentEmbodimentState.initialize(invalid)


@pytest.mark.parametrize("field", _CONTINUOUS_FIELDS)
def test_initialize_rejects_float16_physics(field):
    state = _state((2,))
    value = getattr(state, field).to(torch.float16)
    invalid = _state_with_field_changed(state, field, value)

    with pytest.raises(TypeError, match="float32 or torch.float64"):
        AgentEmbodimentState.initialize(invalid)


@pytest.mark.parametrize("bad_input", [None, object(), "not an AgentState"])
def test_initialize_rejects_invalid_physics_input_type(bad_input):
    with pytest.raises(TypeError, match="AgentState"):
        AgentEmbodimentState.initialize(bad_input)


@pytest.mark.parametrize("field", _CONTINUOUS_FIELDS)
@pytest.mark.parametrize("bad_value", [float("nan"), float("-inf")])
def test_physical_replacement_rejects_nonfinite_values(field, bad_value):
    embodiment = AgentEmbodimentState.initialize(_state((2,)))
    replacement = embodiment.physics.clone()
    value = getattr(replacement, field).clone()
    value.reshape(-1)[0] = bad_value
    replacement = _state_with_field_changed(replacement, field, value)

    with pytest.raises(ValueError, match="finite"):
        embodiment.replace_physics(replacement)


@pytest.mark.parametrize("field", _CONTINUOUS_FIELDS)
def test_physical_replacement_rejects_unsupported_dtypes(field):
    embodiment = AgentEmbodimentState.initialize(_state((2,)))
    replacement = embodiment.physics.clone()
    value = getattr(replacement, field).to(torch.int64)
    replacement = _state_with_field_changed(replacement, field, value)

    with pytest.raises(TypeError, match="float32 or torch.float64"):
        embodiment.replace_physics(replacement)


@pytest.mark.parametrize("bad_input", [None, object(), "not an AgentState"])
def test_physical_replacement_rejects_invalid_input_type(bad_input):
    embodiment = AgentEmbodimentState.initialize(_state((2,)))

    with pytest.raises(TypeError, match="AgentState"):
        embodiment.replace_physics(bad_input)


def test_zero_energy_alive_state_is_valid_until_p0_step():
    # P0 deliberately permits this initial state. The embodiment wrapper
    # must not derive alive from energy or pre-empt P0's transition.
    physics = make_state((1,), energy=0.0)
    assert physics.alive.item()
    assert physics.energy.item() == 0.0

    embodiment = AgentEmbodimentState.initialize(physics)
    assert embodiment.physics.alive.item()
    assert not embodiment.failed.item()
    assert not embodiment.extracted.item()

    next_physics, diagnostics = step(
        embodiment.physics,
        actions=torch.zeros((1, 2), dtype=physics.position.dtype),
        gravity=torch.zeros((2,), dtype=physics.position.dtype),
        config=PhysicsConfig(),
    )

    # The P0 step owns the exhaustion transition.
    assert not next_physics.alive.item()
    assert next_physics.energy.item() == 0.0

    updated = embodiment.replace_physics(next_physics)
    assert not updated.physics.alive.item()
    assert not updated.failed.item()
    assert not updated.extracted.item()


# Cross-device lifecycle, replacement, and measurement coverage.

_CROSS_DEVICE_CASES = [
    pytest.param("cpu", torch.float32, (3,), id="cpu-f32-N"),
    pytest.param("cpu", torch.float64, (3,), id="cpu-f64-N"),
    pytest.param("cpu", torch.float32, (2, 2), id="cpu-f32-BN"),
    pytest.param("cpu", torch.float64, (2, 2), id="cpu-f64-BN"),
    pytest.param(
        "cuda", torch.float32, (3,),
        id="cuda-f32-N",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
    pytest.param(
        "cuda", torch.float64, (3,),
        id="cuda-f64-N",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
    pytest.param(
        "cuda", torch.float32, (2, 2),
        id="cuda-f32-BN",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
    pytest.param(
        "cuda", torch.float64, (2, 2),
        id="cuda-f64-BN",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA is not available"
        ),
    ),
]


def _cross_device_state(shape, device, dtype, *, energy=100.0):
    return make_state(
        shape,
        device=device,
        dtype=dtype,
        energy=energy,
    )


def _cross_device_mask(shape, device, selected_flat_indices):
    mask = torch.zeros(
        shape + (1,), dtype=torch.bool, device=device
    )
    mask.reshape(-1)[list(selected_flat_indices)] = True
    return mask


@pytest.mark.parametrize(
    "device,dtype,shape", _CROSS_DEVICE_CASES
)
def test_cross_device_terminal_annotations_idempotence_and_conflicts(
    device, dtype, shape
):
    embodiment = AgentEmbodimentState.initialize(
        _cross_device_state(shape, device, dtype)
    )
    count = embodiment.physics.alive.numel()
    assert count >= 2

    extracted_mask = _cross_device_mask(shape, device, [0])
    failed_mask = _cross_device_mask(shape, device, [1])

    extracted = embodiment.mark_extracted(extracted_mask)
    failed = extracted.mark_failed(failed_mask)

    assert torch.equal(
        failed.extracted.reshape(-1)[:2],
        torch.tensor([True, False], device=device),
    )
    assert torch.equal(
        failed.failed.reshape(-1)[:2],
        torch.tensor([False, True], device=device),
    )
    assert not failed.physics.alive.reshape(-1)[:2].any()

    repeated = failed.mark_extracted(extracted_mask)
    repeated = repeated.mark_failed(failed_mask)
    assert torch.equal(repeated.extracted, failed.extracted)
    assert torch.equal(repeated.failed, failed.failed)
    assert torch.equal(repeated.physics.alive, failed.physics.alive)

    with pytest.raises(ValueError):
        failed.mark_failed(extracted_mask)

    with pytest.raises(ValueError):
        failed.mark_extracted(failed_mask)


@pytest.mark.parametrize(
    "device,dtype,shape", _CROSS_DEVICE_CASES
)
def test_cross_device_replacement_preserves_frozen_state_and_blocks_resurrection(
    device, dtype, shape
):
    embodiment = AgentEmbodimentState.initialize(
        _cross_device_state(shape, device, dtype)
    )

    failed_mask = _cross_device_mask(shape, device, [0])
    terminal = embodiment.mark_failed(failed_mask)

    # A live agent may receive a valid physical transition.
    replacement = terminal.physics.clone()
    active = terminal.physics.alive.expand_as(terminal.physics.position)
    replacement.position = torch.where(
        active,
        terminal.physics.position + 0.25,
        terminal.physics.position,
    )
    updated = terminal.replace_physics(replacement)

    assert torch.equal(updated.physics.position, replacement.position)
    assert not updated.physics.alive.reshape(-1)[0]

    # But changing any already-dead agent's physical state is forbidden.
    moved_dead = updated.physics.clone()
    moved_dead.position.reshape(-1, 2)[0, 0] += 1.0
    with pytest.raises(ValueError, match="frozen physical state"):
        updated.replace_physics(moved_dead)

    # Nor may a replacement reactivate an inactive agent.
    resurrected_alive = updated.physics.alive.clone()
    resurrected_alive.reshape(-1)[0] = True
    resurrected = AgentState(
        position=updated.physics.position,
        velocity=updated.physics.velocity,
        angle=updated.physics.angle,
        angular_velocity=updated.physics.angular_velocity,
        energy=updated.physics.energy,
        alive=resurrected_alive,
    )
    with pytest.raises(ValueError, match="reactivate"):
        updated.replace_physics(resurrected)


@pytest.mark.parametrize(
    "device,dtype,shape", _CROSS_DEVICE_CASES
)
def test_cross_device_measurements_store_explicit_finite_difference_and_own_inputs(
    device, dtype, shape
):
    dt = 1.0 / 60.0
    physics = _cross_device_state(shape, device, dtype)
    embodiment = AgentEmbodimentState.initialize(physics)

    next_physics = physics.clone()
    delta_velocity = torch.full_like(
        next_physics.velocity, 0.3
    )
    next_physics.velocity = physics.velocity + delta_velocity

    # Test-side provenance: velocity difference over the completed step.
    measured = (next_physics.velocity - physics.velocity) / dt
    applied_actions = torch.full(
        shape + (2,), 0.25, device=device, dtype=dtype
    )

    updated = embodiment.record_transition_measurements(
        applied_actions=applied_actions,
        measured_acceleration=measured,
    )

    expected = delta_velocity / dt
    assert torch.allclose(
        updated.measured_acceleration, expected, rtol=1e-6, atol=1e-6
    )
    assert torch.equal(updated.previous_actions, applied_actions)

    # The wrapper must own storage, not retain caller-owned input tensors.
    applied_actions.fill_(0.0)
    measured.fill_(0.0)
    assert torch.all(updated.previous_actions == 0.25)
    assert torch.allclose(
        updated.measured_acceleration, expected, rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize(
    "device,dtype,shape", _CROSS_DEVICE_CASES
)
def test_cross_device_zero_energy_is_transitioned_by_p0_step(
    device, dtype, shape
):
    physics = _cross_device_state(
        shape, device, dtype, energy=0.0
    )
    assert physics.alive.all()

    embodiment = AgentEmbodimentState.initialize(physics)
    assert embodiment.physics.alive.all()
    assert not embodiment.failed.any()
    assert not embodiment.extracted.any()

    actions = torch.zeros(
        shape + (2,), device=device, dtype=dtype
    )
    gravity = torch.zeros((2,), device=device, dtype=dtype)

    next_physics, diagnostics = step(
        embodiment.physics,
        actions=actions,
        gravity=gravity,
        config=PhysicsConfig(),
    )

    assert not next_physics.alive.any()
    assert torch.all(next_physics.energy == 0.0)

    updated = embodiment.replace_physics(next_physics)
    assert not updated.physics.alive.any()
    assert not updated.failed.any()
    assert not updated.extracted.any()
