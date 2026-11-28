from typing import Optional


import torch
from torch import Tensor, nn


from .interface import Controller




class StaticController(Controller):
    """Seeded stateless 23 -> 32 -> 32 -> 2 tanh controller."""


    def __init__(self, seed: int = 7, dtype=torch.float32):
        super().__init__()
        self.seed = int(seed)


        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.network = nn.Sequential(
                nn.Linear(23, 32, dtype=dtype),
                nn.Tanh(),
                nn.Linear(32, 32, dtype=dtype),
                nn.Tanh(),
                nn.Linear(32, 2, dtype=dtype),
                nn.Tanh(),
            )


        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)


        with torch.no_grad():
            for layer in self.network:
                if isinstance(layer, nn.Linear):
                    bound = layer.weight.shape[1] ** -0.5
                    layer.weight.uniform_(
                        -bound, bound, generator=generator
                    )
                    layer.bias.uniform_(
                        -bound, bound, generator=generator
                    )


    def act(
        self,
        observations: Tensor,
        active_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if observations.ndim != 3:
            raise ValueError("observations must have shape [B,N,23]")
        if observations.shape[-1] != self.observation_dim:
            raise ValueError("observation dimension must be 23")


        actions = self.network(observations)


        if active_mask is not None:
            if active_mask.shape != observations.shape[:2]:
                raise ValueError("active_mask must have shape [B,N]")
            mask = active_mask.to(
                device=actions.device, dtype=actions.dtype
            ).unsqueeze(-1)
            actions = actions * mask


        return actions


    def reset_episode(self) -> None:
        """STATIC has no episode-local state."""
        return None

