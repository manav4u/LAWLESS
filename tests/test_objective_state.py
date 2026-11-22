import pytest
import torch

from lawless.world import AgentCoreCollectionState


def test_initialize_creates_independent_boolean_masks():
    state = AgentCoreCollectionState.initialize((2, 3), 4)

    assert state.collected.shape == (2, 3, 4)
    assert state.collected.dtype is torch.bool
    assert state.collected.device.type == "cpu"
    assert not state.collected.any()


def test_collection_is_independent_per_agent_and_allows_multiple_cores():
    state = AgentCoreCollectionState.initialize((2,), 3)

    updated, newly = state.mark_collected(
        torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        torch.tensor([0, 2, 0, 1], dtype=torch.int64),
    )

    assert newly.tolist() == [True, True, True, True]
    assert updated.collected.tolist() == [
        [True, False, True],
        [True, True, False],
    ]
    assert not state.collected.any()


def test_same_core_can_be_collected_independently_by_every_agent():
    state = AgentCoreCollectionState.initialize((3,), 1)

    updated, newly = state.mark_collected(
        torch.tensor([0, 1, 2], dtype=torch.int64),
        torch.tensor([0, 0, 0], dtype=torch.int64),
    )

    assert newly.tolist() == [True, True, True]
    assert updated.collected[:, 0].tolist() == [True, True, True]


def test_repeated_pair_is_idempotent_within_one_event_batch():
    state = AgentCoreCollectionState.initialize((2,), 3)

    updated, newly = state.mark_collected(
        torch.tensor([0, 0, 1, 0, 1], dtype=torch.int64),
        torch.tensor([2, 2, 1, 1, 1], dtype=torch.int64),
    )

    # First occurrence changes state; later occurrences are explicit no-ops.
    assert newly.tolist() == [True, False, True, True, False]
    assert updated.collected.tolist() == [
        [False, True, True],
        [False, True, False],
    ]


def test_repeated_event_against_existing_state_is_noop():
    state = AgentCoreCollectionState.initialize((2,), 2)
    state, first_newly = state.mark_collected(
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
    )
    assert first_newly.tolist() == [True]

    repeated, newly = state.mark_collected(
        torch.tensor([1, 1], dtype=torch.int64),
        torch.tensor([0, 0], dtype=torch.int64),
    )

    assert newly.tolist() == [False, False]
    assert torch.equal(repeated.collected, state.collected)
    assert repeated is not state


def test_empty_event_batch_is_a_valid_noop():
    state = AgentCoreCollectionState.initialize((2, 3), 4)

    updated, newly = state.mark_collected(
        torch.empty((0,), dtype=torch.int64),
        torch.empty((0,), dtype=torch.int64),
    )

    assert newly.shape == (0,)
    assert newly.dtype is torch.bool
    assert torch.equal(updated.collected, state.collected)
    assert updated is not state


def test_zero_core_layout_is_supported():
    state = AgentCoreCollectionState.initialize((2, 3), 0)

    assert state.collected.shape == (2, 3, 0)
    assert state.collected.numel() == 0

    updated, newly = state.mark_collected(
        torch.empty((0,), dtype=torch.int64),
        torch.empty((0,), dtype=torch.int64),
    )

    assert updated.collected.shape == (2, 3, 0)
    assert newly.shape == (0,)


def test_zero_agent_batch_is_supported():
    state = AgentCoreCollectionState.initialize((0, 2), 3)

    assert state.collected.shape == (0, 2, 3)

    updated, newly = state.mark_collected(
        torch.empty((0,), dtype=torch.int64),
        torch.empty((0,), dtype=torch.int64),
    )

    assert updated.collected.shape == (0, 2, 3)
    assert newly.shape == (0,)


def test_scalar_batch_shape_represents_one_agent():
    state = AgentCoreCollectionState.initialize((), 2)

    assert state.collected.shape == (2,)

    updated, newly = state.mark_collected(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )

    assert newly.tolist() == [True]
    assert updated.collected.tolist() == [False, True]


def test_constructor_owns_input_tensor_storage():
    source = torch.zeros((2, 3), dtype=torch.bool)
    state = AgentCoreCollectionState(source)

    source[0, 1] = True

    assert not state.collected[0, 1]


def test_clone_owns_independent_tensor_storage():
    state = AgentCoreCollectionState.initialize((2,), 2)
    cloned = state.clone()

    assert cloned is not state
    assert cloned.collected.data_ptr() != state.collected.data_ptr()

    updated, _ = cloned.mark_collected(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )

    assert not state.collected.any()
    assert updated.collected[0, 1]


def test_event_inputs_are_not_retained_or_mutated():
    state = AgentCoreCollectionState.initialize((2,), 2)
    agents = torch.tensor([0, 1], dtype=torch.int64)
    cores = torch.tensor([1, 0], dtype=torch.int64)

    updated, newly = state.mark_collected(agents, cores)
    agents.fill_(0)
    cores.fill_(0)

    assert newly.tolist() == [True, True]
    assert updated.collected.tolist() == [
        [False, True],
        [True, False],
    ]


@pytest.mark.parametrize(
    "bad_mask",
    [
        None,
        [[False, True]],
        torch.zeros((2, 2), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.int64),
    ],
)
def test_constructor_rejects_invalid_collection_mask(bad_mask):
    with pytest.raises((TypeError, ValueError)):
        AgentCoreCollectionState(bad_mask)


@pytest.mark.parametrize(
    "bad_shape",
    [
        None,
        [2, 3],
        (2, -1),
        (2, 1.5),
        (2, True),
    ],
)
def test_initialize_rejects_invalid_batch_shape(bad_shape):
    with pytest.raises((TypeError, ValueError)):
        AgentCoreCollectionState.initialize(bad_shape, 3)


@pytest.mark.parametrize("bad_count", [-1, 1.5, True, None])
def test_initialize_rejects_invalid_core_count(bad_count):
    with pytest.raises((TypeError, ValueError)):
        AgentCoreCollectionState.initialize((2,), bad_count)


@pytest.mark.parametrize(
    "bad_agents",
    [
        [0],
        torch.tensor(0, dtype=torch.int64),
        torch.tensor([[0]], dtype=torch.int64),
        torch.tensor([0.0]),
        torch.tensor([True]),
    ],
)
def test_mark_collected_rejects_invalid_agent_indices(bad_agents):
    state = AgentCoreCollectionState.initialize((2,), 2)

    with pytest.raises((TypeError, ValueError)):
        state.mark_collected(
            bad_agents,
            torch.tensor([0], dtype=torch.int64),
        )


@pytest.mark.parametrize(
    "bad_cores",
    [
        [0],
        torch.tensor(0, dtype=torch.int64),
        torch.tensor([[0]], dtype=torch.int64),
        torch.tensor([0.0]),
        torch.tensor([True]),
    ],
)
def test_mark_collected_rejects_invalid_core_indices(bad_cores):
    state = AgentCoreCollectionState.initialize((2,), 2)

    with pytest.raises((TypeError, ValueError)):
        state.mark_collected(
            torch.tensor([0], dtype=torch.int64),
            bad_cores,
        )


def test_mark_collected_rejects_mismatched_event_shapes():
    state = AgentCoreCollectionState.initialize((2,), 2)

    with pytest.raises(ValueError, match="equal shape"):
        state.mark_collected(
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
        )


@pytest.mark.parametrize(
    ("agents", "cores", "message"),
    [
        ([-1], [0], "agent index"),
        ([2], [0], "agent index"),
        ([0], [-1], "core index"),
        ([0], [2], "core index"),
    ],
)
def test_mark_collected_rejects_out_of_range_indices(
    agents, cores, message
):
    state = AgentCoreCollectionState.initialize((2,), 2)

    with pytest.raises(IndexError, match=message):
        state.mark_collected(
            torch.tensor(agents, dtype=torch.int64),
            torch.tensor(cores, dtype=torch.int64),
        )

    # Invalid batches are rejected atomically.
    assert not state.collected.any()


def test_mark_collected_rejects_wrong_index_device():
    state = AgentCoreCollectionState.initialize((2,), 2)

    if torch.cuda.is_available():
        wrong_device = "cuda"
    else:
        wrong_device = "meta"

    agents = torch.tensor([0], dtype=torch.int64, device=wrong_device)
    cores = torch.tensor([0], dtype=torch.int64, device=wrong_device)

    with pytest.raises(ValueError, match="device"):
        state.mark_collected(agents, cores)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_state_dtype_is_boolean_independent_of_physics_precision(dtype):
    # Objective bookkeeping has no floating-point energy/reward balance.
    state = AgentCoreCollectionState.initialize((2, 3), 4)

    assert state.collected.dtype is torch.bool
    assert state.collected.dtype != dtype

# Append to tests/test_p1_objective_state.py

import pytest
import torch

from lawless.world.objective_state import AgentCoreCollectionState


CUDA_CASE = pytest.param(
    "cuda",
    marks=pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA is not available",
    ),
)


def _cpu_collection_reference(
    initial_mask: torch.Tensor,
    agent_indices: torch.Tensor,
    core_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deliberately simple ordered reference; used only by tests."""
    expected = initial_mask.clone()
    newly = torch.zeros(agent_indices.numel(), dtype=torch.bool)

    flat = expected.reshape(-1)
    core_count = expected.shape[-1]
    seen_in_batch: set[tuple[int, int]] = set()

    for event_index, (agent, core) in enumerate(
        zip(agent_indices.tolist(), core_indices.tolist())
    ):
        pair = (agent, core)
        if pair in seen_in_batch:
            continue
        seen_in_batch.add(pair)

        key = agent * core_count + core
        if not bool(flat[key]):
            newly[event_index] = True
            flat[key] = True

    return expected, newly


@pytest.mark.parametrize("device", ["cpu", CUDA_CASE])
def test_mark_collected_ordered_duplicates_match_cpu_reference(device):
    device = torch.device(device)
    initial = torch.zeros((2, 3, 5), dtype=torch.bool)
    initial[0, 1, 2] = True
    initial[1, 0, 4] = True

    agents = torch.tensor(
        [5, 1, 5, 0, 1, 5, 2, 0, 5, 1, 2, 5],
        dtype=torch.int64,
    )
    cores = torch.tensor(
        [3, 2, 3, 4, 2, 1, 0, 4, 3, 0, 0, 1],
        dtype=torch.int64,
    )

    expected_mask, expected_new = _cpu_collection_reference(
        initial, agents, cores
    )

    state = AgentCoreCollectionState(initial.to(device))
    result, newly = state.mark_collected(
        agents.to(device),
        cores.to(device),
    )

    assert torch.equal(result.collected.cpu(), expected_mask)
    assert torch.equal(newly.cpu(), expected_new)

    # The first occurrence wins even when a later event repeats the pair.
    assert newly.cpu().tolist() == [
        True, False, False, True, False, True,
        True, False, False, True, False, False,
    ]

    # Neither input state nor event tensors were mutated.
    assert torch.equal(state.collected.cpu(), initial)
    assert torch.equal(agents, torch.tensor(
        [5, 1, 5, 0, 1, 5, 2, 0, 5, 1, 2, 5],
        dtype=torch.int64,
    ))


@pytest.mark.parametrize("device", ["cpu", CUDA_CASE])
def test_mark_collected_multiaxis_batch_matches_cpu_reference(device):
    device = torch.device(device)
    initial = torch.zeros((2, 2, 3, 7), dtype=torch.bool)
    initial[0, 1, 2, 5] = True
    initial[1, 0, 1, 1] = True

    # Flattened agent indices address the (2, 2, 3) batch dimensions.
    agents = torch.tensor(
        [11, 0, 7, 11, 3, 7, 0, 11, 5, 3, 5, 11, 0],
        dtype=torch.int64,
    )
    cores = torch.tensor(
        [6, 2, 4, 6, 1, 4, 2, 0, 3, 1, 3, 0, 2],
        dtype=torch.int64,
    )

    expected_mask, expected_new = _cpu_collection_reference(
        initial, agents, cores
    )

    state = AgentCoreCollectionState(initial.to(device))
    result, newly = state.mark_collected(
        agents.to(device),
        cores.to(device),
    )

    assert torch.equal(result.collected.cpu(), expected_mask)
    assert torch.equal(newly.cpu(), expected_new)


@pytest.mark.parametrize("device", ["cpu", CUDA_CASE])
def test_mark_collected_previously_collected_pairs_are_not_new(device):
    device = torch.device(device)
    initial = torch.zeros((2, 4), dtype=torch.bool)
    initial[0, 1] = True
    initial[1, 3] = True

    agents = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64)
    cores = torch.tensor([1, 3, 1, 0, 2, 3], dtype=torch.int64)

    expected_mask, expected_new = _cpu_collection_reference(
        initial, agents, cores
    )

    state = AgentCoreCollectionState(initial.to(device))
    result, newly = state.mark_collected(
        agents.to(device),
        cores.to(device),
    )

    assert torch.equal(result.collected.cpu(), expected_mask)
    assert torch.equal(newly.cpu(), expected_new)
    assert newly.cpu().tolist() == [False, False, False, True, True, False]


@pytest.mark.parametrize("device", ["cpu", CUDA_CASE])
def test_mark_collected_empty_events_preserve_mask_and_return_empty_output(device):
    device = torch.device(device)
    initial = torch.tensor(
        [[True, False, False], [False, True, False]],
        dtype=torch.bool,
    )
    state = AgentCoreCollectionState(initial.to(device))

    result, newly = state.mark_collected(
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.empty((0,), dtype=torch.int64, device=device),
    )

    assert torch.equal(result.collected.cpu(), initial)
    assert newly.dtype == torch.bool
    assert newly.device == device
    assert newly.shape == (0,)
    assert result.collected.data_ptr() != state.collected.data_ptr()


@pytest.mark.parametrize("device", ["cpu", CUDA_CASE])
def test_mark_collected_zero_core_accepts_only_empty_events(device):
    device = torch.device(device)
    state = AgentCoreCollectionState.initialize(
        (2, 3), 0, device=device
    )

    result, newly = state.mark_collected(
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.empty((0,), dtype=torch.int64, device=device),
    )

    assert result.collected.shape == (2, 3, 0)
    assert newly.shape == (0,)
    assert newly.dtype == torch.bool
    assert torch.equal(result.collected.cpu(), state.collected.cpu())

    with pytest.raises(IndexError):
        state.mark_collected(
            torch.tensor([0], dtype=torch.int64, device=device),
            torch.tensor([0], dtype=torch.int64, device=device),
        )


@pytest.mark.parametrize("device", ["cpu", CUDA_CASE])
def test_mark_collected_zero_agent_batch_accepts_only_empty_events(device):
    device = torch.device(device)
    state = AgentCoreCollectionState.initialize(
        (0, 2), 4, device=device
    )

    result, newly = state.mark_collected(
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.empty((0,), dtype=torch.int64, device=device),
    )

    assert result.collected.shape == (0, 2, 4)
    assert newly.shape == (0,)
    assert torch.equal(result.collected.cpu(), state.collected.cpu())

    with pytest.raises(IndexError):
        state.mark_collected(
            torch.tensor([0], dtype=torch.int64, device=device),
            torch.tensor([0], dtype=torch.int64, device=device),
        )


@pytest.mark.parametrize("device", ["cpu", CUDA_CASE])
def test_mark_collected_large_batch_matches_ordered_cpu_reference(device):
    device = torch.device(device)
    agent_count = 48
    core_count = 31
    event_count = 12_000

    generator = torch.Generator(device="cpu").manual_seed(7319)
    initial = torch.rand(
        (agent_count, core_count),
        generator=generator,
    ) < 0.12

    # Deliberately draw from a much smaller key space than the event count,
    # producing many duplicates with meaningful first-occurrence ordering.
    agents = torch.randint(
        agent_count, (event_count,), generator=generator, dtype=torch.int64
    )
    cores = torch.randint(
        core_count, (event_count,), generator=generator, dtype=torch.int64
    )

    expected_mask, expected_new = _cpu_collection_reference(
        initial, agents, cores
    )

    state = AgentCoreCollectionState(initial.to(device))
    result, newly = state.mark_collected(
        agents.to(device),
        cores.to(device),
    )

    assert torch.equal(result.collected.cpu(), expected_mask)
    assert torch.equal(newly.cpu(), expected_new)
    assert int(newly.sum().item()) == int(expected_new.sum().item())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_mark_collected_rejects_cross_device_event_indices():
    cuda_state = AgentCoreCollectionState.initialize(
        (2, 3), 4, device="cuda"
    )
    cpu_indices = torch.tensor([0], dtype=torch.int64)

    with pytest.raises(ValueError, match="collection-state device"):
        cuda_state.mark_collected(
            cpu_indices,
            cpu_indices,
        )

    cpu_state = AgentCoreCollectionState.initialize(
        (2, 3), 4, device="cpu"
    )
    cuda_indices = torch.tensor([0], dtype=torch.int64, device="cuda")

    with pytest.raises(ValueError, match="collection-state device"):
        cpu_state.mark_collected(
            cuda_indices,
            cuda_indices,
        )