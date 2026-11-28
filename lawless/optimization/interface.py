from abc import ABC, abstractmethod
import torch


class Optimizer(ABC):
    """Optimizer-independent ask/evaluate/tell boundary."""


    def __init__(self, seed=7):
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(int(seed))
        self.best_vector = None
        self.best_fitness = float("-inf")


    @abstractmethod
    def ask(self):
        """Return a list of candidate parameter vectors."""

    @abstractmethod
    def tell(self, candidates, fitnesses):
        """Consume candidates and their scalar fitnesses."""


    def evaluate(self, evaluator, candidates):
        return evaluator.evaluate_many(candidates)


    def update_best(self, candidates, fitnesses):
        for vector, score in zip(candidates, fitnesses):
            if score > self.best_fitness:
                self.best_fitness = float(score)
                self.best_vector = vector.detach().cpu().clone()

