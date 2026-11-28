from collections.abc import Callable, Sequence


import torch
from torch import Tensor


def flatten_parameters(controller) -> Tensor:
    return torch.cat([p.detach().cpu().reshape(-1)
                      for p in controller.parameters()])


def load_parameters(controller, vector: Tensor) -> None:
    offset = 0
    with torch.no_grad():
        for p in controller.parameters():
            count = p.numel()
            p.copy_(vector[offset:offset+count].reshape(p.shape))
            offset += count
    if offset != vector.numel():
        raise ValueError("parameter vector length mismatch")

class CandidateEvaluator:
    """Evaluate a controller on fixed, single-slot episode seeds."""


    def __init__(self, controller_factory: Callable, env_factory: Callable,
                 seeds: Sequence[int], max_steps: int):
        if not seeds or max_steps <= 0:
            raise ValueError("seeds and positive max_steps required")
        self.controller_factory = controller_factory
        self.env_factory = env_factory
        self.seeds = tuple(int(s) for s in seeds)
        self.max_steps = int(max_steps)


    def evaluate(self, vector: Tensor) -> float:
        controller = self.controller_factory()
        load_parameters(controller, vector)
        successes = []

        for seed in self.seeds:
            env = self.env_factory()
            controller.reset_episode()
            obs = env.reset(seed=seed)
            if obs.shape != (1, 1, 23):
                raise ValueError("evaluator requires observations [1,1,23]")
            success = False
            for _ in range(self.max_steps):
                actions = controller.act(obs)
                result = env.step(actions)
                env, obs, reward, terminated, truncated, info = result
                success = success or bool(reward.reshape(-1)[0].item())
                done = bool(terminated.reshape(-1)[0].item()) or \
                       bool(truncated.reshape(-1)[0].item())
                if done:
                    break

            successes.append(float(success))
        return sum(successes) / len(successes)


    def evaluate_many(self, candidates):
        return [self.evaluate(candidate) for candidate in candidates]

