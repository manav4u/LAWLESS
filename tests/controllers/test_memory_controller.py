import torch
import pytest


from lawless.controllers import MemoryController




def inputs(batch=2, slots=3):
    return torch.randn(batch, slots, 23)




def test_shape_and_bounded_actions():
    controller = MemoryController()
    obs = inputs()
    actions = controller.act(obs)
    assert actions.shape == (2, 3, 2)
    assert torch.isfinite(actions).all()
    assert (actions.abs() <= 1).all()
    assert controller.hidden_state.shape == (2, 3, 32)

def test_initial_state_is_zero():
    controller = MemoryController()
    state = controller.initial_state(inputs())
    assert state.shape == (2, 3, 32)
    assert torch.count_nonzero(state) == 0




def test_seeded_initialization():
    a = MemoryController(seed=7)
    b = MemoryController(seed=7)
    c = MemoryController(seed=8)
    pa = list(a.parameters())
    pb = list(b.parameters())
    pc = list(c.parameters())
    assert all(torch.equal(x, y) for x, y in zip(pa, pb))
    assert any(not torch.equal(x, y) for x, y in zip(pa, pc))

def test_global_rng_unchanged():
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    MemoryController(seed=7)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)




def test_reset_episode_clears_state():
    controller = MemoryController()
    controller.act(inputs())
    assert controller.hidden_state is not None
    controller.reset_episode()
    assert controller.hidden_state is None
    controller.act(inputs())
    assert controller.hidden_state.shape == (2, 3, 32)

def test_masked_slots_emit_zero_and_freeze_state():
    controller = MemoryController()
    obs = inputs()
    controller.act(obs)
    before = controller.hidden_state.clone()
    mask = torch.tensor([[True, False, True],
                         [False, True, False]])
    actions = controller.act(obs, active_mask=mask)
    assert torch.count_nonzero(actions[~mask]) == 0
    assert torch.equal(controller.hidden_state[~mask], before[~mask])
    assert torch.isfinite(controller.hidden_state).all()

def test_slots_have_independent_state():
    controller = MemoryController()
    obs = inputs()
    controller.act(obs)
    state = controller.hidden_state.clone()
    assert not torch.equal(state[:, 0], state[:, 1])




def test_invalid_shapes():
    controller = MemoryController()
    with pytest.raises(ValueError):
        controller.act(torch.randn(2, 23))
    with pytest.raises(ValueError):
        controller.act(inputs(), active_mask=torch.ones(2, 2))

