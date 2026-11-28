import math
import torch
from .interface import Optimizer




class CMAES(Optimizer):
    """CPU covariance-matrix adaptation evolution strategy."""


    def __init__(self, initial, population=None, sigma=0.1, seed=7):
        super().__init__(seed)
        self.center = initial.detach().cpu().clone().reshape(-1)
        if self.center.numel() < 1 or not torch.isfinite(self.center).all():
            raise ValueError("initial must be a finite nonempty vector")
        self.dim = self.center.numel()
        if sigma <= 0 or not math.isfinite(float(sigma)):
            raise ValueError("sigma must be finite and positive")
        self.sigma = float(sigma)
        self.population_size = population or (4 + int(3 * math.log(self.dim)))

        if self.population_size < 2:
            raise ValueError("population must be >= 2")
        self.mu = self.population_size // 2
        weights = torch.log(torch.arange(
            1, self.mu + 1, dtype=torch.float64
        ) + 0.5).flip(0)
        self.weights = weights / weights.sum()
        self.mueff = 1.0 / self.weights.square().sum().item()
        n = self.dim
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)

        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff)
            / ((n + 2) ** 2 + self.mueff),
        )
        self.damps = 1 + 2 * max(0, math.sqrt(
            (self.mueff - 1) / (n + 1)
        ) - 1) + self.cs
        self.chi_n = math.sqrt(n) * (
            1 - 1 / (4 * n) + 1 / (21 * n * n)
        )
        self.pc = torch.zeros(n, dtype=torch.float64)
        self.ps = torch.zeros(n, dtype=torch.float64)
        self.cov = torch.eye(n, dtype=torch.float64)
        self.eigenvalues = torch.ones(n, dtype=torch.float64)
        self.basis = torch.eye(n, dtype=torch.float64)
        self.generation = 0
        self.pending = None

    def ask(self):
        if self.pending is not None:
            return [v.clone() for v in self.pending["candidates"]]
        z = torch.randn(
            (self.population_size, self.dim), generator=self.generator,
            dtype=torch.float64,
        )
        transform = self.basis * self.eigenvalues.sqrt().unsqueeze(0)
        y = z @ transform.T
        candidates = self.center.double() + self.sigma * y
        self.pending = {"z": z, "y": y, "candidates": candidates}
        return [v.clone() for v in candidates]

    def tell(self, candidates, fitnesses):
        if self.pending is None:
            raise RuntimeError("tell called before ask")
        if len(candidates) != self.population_size:
            raise ValueError("unexpected population size")
        if len(fitnesses) != self.population_size:
            raise ValueError("candidate/fitness length mismatch")
        scores = [float(f) for f in fitnesses]
        if not all(math.isfinite(f) for f in scores):
            raise ValueError("fitnesses must be finite")
        self.update_best(candidates, scores)
        ranked = sorted(range(len(scores)), key=scores.__getitem__,
                       reverse=True)[:self.mu]
        y = self.pending["y"][ranked]
        z = self.pending["z"][ranked]

        w = self.weights
        ymean = (w[:, None] * y).sum(dim=0)
        zmean = (w[:, None] * z).sum(dim=0)
        self.center += self.sigma * ymean.to(self.center.dtype)
        self.ps = ((1 - self.cs) * self.ps
                    + math.sqrt(self.cs * (2 - self.cs) * self.mueff)
                    * (self.basis @ zmean))
        norm_ps = torch.linalg.vector_norm(self.ps).item()
        correction = math.sqrt(1 - (1 - self.cs) ** (2 * (self.generation + 1)))
        hsig = norm_ps / correction / self.chi_n < (
            1.4 + 2 / (self.dim + 1)
        )
        self.pc = ((1 - self.cc) * self.pc
                   + (hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff))
                   * ymean)

        old_cov = self.cov
        rank_mu = sum(
            w[i] * torch.outer(y[i], y[i])
            for i in range(self.mu)
        )
        self.cov = ((1 - self.c1 - self.cmu) * old_cov
                    + self.c1 * (torch.outer(self.pc, self.pc)
                    + (1 - hsig) * self.cc * (2 - self.cc) * old_cov)
                    + self.cmu * rank_mu)
        self.cov = (self.cov + self.cov.T) * 0.5
        self.sigma *= math.exp(
            (self.cs / self.damps) * (norm_ps / self.chi_n - 1)
        )
        self.generation += 1

        if self.generation % max(1, self.dim // 2) == 0:
            eigvals, basis = torch.linalg.eigh(self.cov)
            eigvals = eigvals.clamp_min(1e-14)
            self.eigenvalues = eigvals
            self.basis = basis
            self.cov = basis @ torch.diag(eigvals) @ basis.T
        self.pending = None
