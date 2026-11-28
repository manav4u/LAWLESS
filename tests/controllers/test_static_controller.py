import torch


from lawless.controllers import Controller, StaticController




def test_interface_and_dimensions():
    controller = StaticController(seed=7)
    assert isinstance(controller, Controller)


    x = torch.zeros((3, 4, 23))
    y = controller.act(x)


    assert y.shape == (3, 4, 2)
    assert y.dtype == x.dtype
def test_seeded_initialization():
    a = StaticController(seed=7)
    b = StaticController(seed=7)
    c = StaticController(seed=8)


    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


    assert any(
        not torch.equal(pa, pc)
        for pa, pc in zip(a.parameters(), c.parameters())
    )
def test_global_rng_unchanged():
    torch.manual_seed(123)
    before = torch.random.get_rng_state()


    StaticController(seed=7)


    after = torch.random.get_rng_state()
    assert torch.equal(before, after)




def test_bounded_actions():
    controller = StaticController(seed=7)
    x = torch.randn(2, 3, 23)


    y = controller.act(x)


    assert torch.isfinite(y).all()
    assert torch.all(y >= -1)
    assert torch.all(y <= 1)




def test_inactive_slots_zero():
    controller = StaticController(seed=7)
    x = torch.randn(2, 3, 23)


    mask = torch.tensor([
        [True, False, True],
        [False, False, True],
    ])


    y = controller.act(x, active_mask=mask)
    assert torch.equal(y[0, 1], torch.zeros(2))
    assert torch.equal(y[1, 0], torch.zeros(2))
    assert torch.equal(y[1, 1], torch.zeros(2))




def test_mask_does_not_change_active_outputs():
    controller = StaticController(seed=7)
    x = torch.randn(2, 3, 23)


    y = controller.act(x)
    mask = torch.tensor([
        [True, False, True],
        [False, True, False],
    ])


    masked = controller.act(x, active_mask=mask)


    assert torch.equal(masked[0, 0], y[0, 0])
    assert torch.equal(masked[0, 2], y[0, 2])
    assert torch.equal(masked[1, 1], y[1, 1])
def test_reset_is_stateless():
    controller = StaticController(seed=7)
    x = torch.randn(1, 2, 23)


    before = controller.act(x)
    controller.reset_episode()
    after = controller.act(x)


    assert torch.equal(before, after)




def test_invalid_shapes():
    controller = StaticController(seed=7)


    try:
        controller.act(torch.zeros(23))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid rank accepted")


    try:
        controller.act(torch.zeros(1, 1, 24))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid feature dimension accepted")
    try:
        controller.act(
            torch.zeros(1, 1, 23),
            active_mask=torch.zeros(1, 1, 1),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mask shape accepted")