import torch
import pytest
from lawless.optimization import OpenAIES




def make_es(seed=7):
    return OpenAIES(torch.zeros(5), population=6,
                    sigma=0.1, learning_rate=0.01, seed=seed)




def test_ask_shape_and_antithetic_pairs():
    es = make_es()
    candidates = es.ask()
    assert len(candidates) == 6
    center = es.center
    assert torch.allclose(candidates[0] + candidates[1], 2 * center)
    assert torch.allclose(candidates[2] + candidates[3], 2 * center)
    assert torch.allclose(candidates[4] + candidates[5], 2 * center)

def test_seeded_ask_deterministic():
    a, b = make_es(11), make_es(11)
    assert all(torch.equal(x, y) for x, y in zip(a.ask(), b.ask()))




def test_ask_repeat_before_tell_is_stable():
    es = make_es()
    first, second = es.ask(), es.ask()
    assert all(torch.equal(x, y) for x, y in zip(first, second))




def test_global_rng_unchanged():
    torch.manual_seed(1234)
    before = torch.random.get_rng_state().clone()
    make_es()
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)

def test_tell_updates_center_and_tracks_best():
    es = make_es()
    candidates = es.ask()
    old_center = es.center.clone()
    scores = [0.0, 1.0, 0.5, 0.5, 0.2, 0.2]
    es.tell(candidates, scores)
    assert not torch.equal(es.center, old_center)
    assert es.best_fitness == 1.0
    assert torch.equal(es.best_vector, candidates[1])
    assert es.pending_eps is None

def test_invalid_population_and_tell_lengths():
    with pytest.raises(ValueError):
        OpenAIES(torch.zeros(2), population=3)
    es = make_es()
    candidates = es.ask()
    with pytest.raises(ValueError):
        es.tell(candidates, [1.0])
