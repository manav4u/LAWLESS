import torch
from torch import nn
from lawless.optimization import CandidateEvaluator, GA


class FakeController(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
    def reset_episode(self):
        pass
    def act(self, obs):
        return obs.new_zeros((1, 1, 2))


class FakeEnv:
    def reset(self, seed):
        self.seed = seed
        self.step_count = 0
        return torch.zeros(1, 1, 23)

    def step(self, actions):
        self.step_count += 1
        success = self.seed == 2 and self.step_count == 1
        reward = torch.tensor([[[float(success)]]])
        done = torch.tensor([[True]])
        trunc = torch.tensor([[False]])
        obs = torch.zeros(1, 1, 23)
        return self, obs, reward, done, trunc, {}


def controller_factory():
    return FakeController()


def env_factory():
    return FakeEnv()

def test_fixed_seed_fitness():
    evaluator = CandidateEvaluator(controller_factory, env_factory,
                                  [1, 2, 3], max_steps=4)
    assert evaluator.evaluate(torch.zeros(1)) == 1 / 3


def test_truncation_without_success_scores_zero():
    evaluator = CandidateEvaluator(controller_factory, env_factory,
                                  [1, 3], max_steps=1)
    assert evaluator.evaluate(torch.zeros(1)) == 0.0


def test_seed_repeatability():
    ev = CandidateEvaluator(controller_factory, env_factory,
                            [1, 2, 3], max_steps=4)
    assert ev.evaluate(torch.zeros(1)) == ev.evaluate(torch.zeros(1))

def test_ga_ask_tell_best():
    ga = GA(torch.zeros(4), population=4, seed=7)
    candidates = ga.ask()
    assert len(candidates) == 4
    scores = [0.0, 0.2, 0.5, 0.1]
    ga.tell(candidates, scores)
    assert ga.best_fitness == 0.5
    assert torch.equal(ga.best_vector, candidates[2])
    assert len(ga.ask()) == 4


def test_ga_deterministic_seed():
    a = GA(torch.zeros(4), population=4, seed=7)
    b = GA(torch.zeros(4), population=4, seed=7)
    assert all(torch.equal(x, y) for x, y in zip(a.ask(), b.ask()))

