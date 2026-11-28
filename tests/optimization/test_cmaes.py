import torch
import pytest
from lawless.optimization.cmaes import CMAES




def make_optimizer(seed=7, dim=4, population=6):
    return CMAES(torch.zeros(dim), population=population, seed=seed)




def test_population_shape_and_cpu():
    opt = make_optimizer()
    candidates = opt.ask()
    assert len(candidates) == 6
    assert all(v.shape == (4,) for v in candidates)
    assert all(v.device.type == "cpu" for v in candidates)

def test_ask_is_repeatable_before_tell():
    opt = make_optimizer()
    first = opt.ask()
    second = opt.ask()
    assert all(torch.equal(a, b) for a, b in zip(first, second))




def test_seeded_population_is_deterministic():
    a, b = make_optimizer(19), make_optimizer(19)
    ca, cb = a.ask(), b.ask()
    assert all(torch.equal(x, y) for x, y in zip(ca, cb))

def test_tell_updates_best_and_center():
    opt = make_optimizer()
    candidates = opt.ask()
    fitnesses = [-float(v.square().sum()) for v in candidates]
    opt.tell(candidates, fitnesses)
    assert opt.best_vector is not None
    assert opt.best_fitness == max(fitnesses)
    assert opt.generation == 1
    assert opt.pending is None

def test_multiple_generations_and_best():
    opt = make_optimizer(seed=3)
    for _ in range(4):
        candidates = opt.ask()
        scores = [-float(v.square().sum()) for v in candidates]
        opt.tell(candidates, scores)
    assert opt.generation == 4
    assert torch.isfinite(opt.center).all()
    assert torch.isfinite(opt.cov).all()
    assert opt.best_fitness <= 0

def test_tell_without_ask_raises():
    opt = make_optimizer()
    with pytest.raises(RuntimeError):
        opt.tell([], [])




def test_population_mismatch_raises():
    opt = make_optimizer()
    candidates = opt.ask()
    with pytest.raises(ValueError):
        opt.tell(candidates[:-1], [0.0] * 5)

def test_nonfinite_fitness_rejected():
    opt = make_optimizer()
    candidates = opt.ask()
    with pytest.raises(ValueError):
        opt.tell(candidates, [0.0, 1.0, 2.0, 3.0,
                              float("nan"), 5.0])




def test_invalid_initial_and_sigma():
    with pytest.raises(ValueError):
        CMAES(torch.tensor([float("nan")]))
    with pytest.raises(ValueError):
        CMAES(torch.zeros(3), sigma=0)
