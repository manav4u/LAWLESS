import torch
from .interface import Optimizer




class OpenAIES(Optimizer):
    """Antithetic evolution-strategy optimizer."""


    def __init__(self, initial, population=8, sigma=0.1,
                 learning_rate=0.01, seed=7):
        super().__init__(seed)
        if population < 2 or population % 2 or sigma <= 0:
            raise ValueError("population must be even >=2, sigma > 0")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.center = initial.detach().cpu().clone()
        self.population_size = int(population)
        self.sigma = float(sigma)
        self.learning_rate = float(learning_rate)
        self.pending_eps = None

    def ask(self):
        if self.pending_eps is not None:
            eps = [e.clone() for e in self.pending_eps]
        else:
            eps = []
            for _ in range(self.population_size // 2):
                e = torch.randn(self.center.shape,
                                generator=self.generator)
                eps.extend((e, -e))
            self.pending_eps = eps
        return [self.center + self.sigma * e for e in eps]


    def tell(self, candidates, fitnesses):
        if len(candidates) != self.population_size:
            raise ValueError("unexpected population size")
        if len(fitnesses) != self.population_size:
            raise ValueError("candidate/fitness length mismatch")

        self.update_best(candidates, fitnesses)
        eps = self.pending_eps
        if eps is None:
            raise RuntimeError("tell called before ask")
        gradient = torch.zeros_like(self.center)
        pairs = self.population_size // 2
        for i in range(0, self.population_size, 2):
            delta = float(fitnesses[i]) - float(fitnesses[i + 1])
            gradient += delta * eps[i]
        gradient /= (2.0 * pairs * self.sigma)
        self.center += self.learning_rate * gradient
        self.pending_eps = None

# OpenAIES intentionally uses no external dependencies
# beyond the existing torch-based optimizer interface.
