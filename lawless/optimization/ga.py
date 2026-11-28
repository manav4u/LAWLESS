import torch
from .interface import Optimizer


class GA(Optimizer):
    """Minimal CPU genetic optimizer for continuous parameter vectors."""


    def __init__(self, initial, population=8, sigma=0.1, seed=7):
        super().__init__(seed)
        if population < 2 or sigma <= 0:
            raise ValueError("population >= 2 and sigma > 0 required")
        self.center = initial.detach().cpu().clone()
        self.population_size = int(population)
        self.sigma = float(sigma)
        self.pending = None


    def ask(self):
        if self.pending is not None:
            return [v.clone() for v in self.pending]

        candidates = [self.center.clone()]
        for _ in range(self.population_size - 1):
            noise = torch.randn(self.center.shape,
                                generator=self.generator)
            candidates.append(self.center + self.sigma * noise)
        self.pending = candidates
        return [v.clone() for v in candidates]


    def tell(self, candidates, fitnesses):
        if len(candidates) != len(fitnesses):
            raise ValueError("candidate/fitness length mismatch")
        if len(candidates) != self.population_size:
            raise ValueError("unexpected population size")
        self.update_best(candidates, fitnesses)
        ranked = sorted(range(len(fitnesses)),
                        key=lambda i: fitnesses[i], reverse=True)

        parents = [candidates[i].detach().cpu()
                   for i in ranked[:2]]
        next_pop = [parents[0].clone()]
        for _ in range(self.population_size - 1):
            mask = torch.rand(self.center.shape,
                              generator=self.generator) < 0.5
            child = torch.where(mask, parents[0], parents[1])
            noise = torch.randn(child.shape, generator=self.generator)
            next_pop.append(child + self.sigma * noise)
        self.center = parents[0].clone()
        self.pending = next_pop

