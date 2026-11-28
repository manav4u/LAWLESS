from typing import Optional


import torch
from torch import Tensor, nn


from .interface import Controller




class MemoryController(Controller):
    """Elman controller: 23 inputs, 32 hidden, 2 tanh actions."""


    hidden_dim = 32


    def __init__(self, seed: int = 7, dtype=torch.float32):
        super().__init__()
        self.seed = int(seed)
        self.hidden_state: Optional[Tensor] = None

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.recurrent = nn.Linear(23 + 32, 32, dtype=dtype)
            self.output = nn.Linear(32, 2, dtype=dtype)


        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        with torch.no_grad():
            for layer in (self.recurrent, self.output):
                bound = layer.weight.shape[1] ** -0.5
                layer.weight.uniform_(-bound, bound, generator=generator)
                layer.bias.uniform_(-bound, bound, generator=generator)

    def initial_state(self, observations: Tensor) -> Tensor:
        if observations.ndim != 3 or observations.shape[-1] != 23:
            raise ValueError("observations must have shape [B,N,23]")
        return observations.new_zeros(
            (*observations.shape[:2], self.hidden_dim)
        )


    def act(
        self,
        observations: Tensor,
        active_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if observations.ndim != 3 or observations.shape[-1] != 23:
            raise ValueError("observations must have shape [B,N,23]")
        batch_shape = observations.shape[:2]

        if self.hidden_state is None:
            previous = self.initial_state(observations)
        else:
            if self.hidden_state.shape != (*batch_shape, 32):
                raise ValueError("hidden_state must have shape [B,N,32]")
            previous = self.hidden_state.to(
                device=observations.device, dtype=observations.dtype
            )


        combined = torch.cat((observations, previous), dim=-1)
        next_hidden = torch.tanh(self.recurrent(combined))
        actions = torch.tanh(self.output(next_hidden))

        if active_mask is None:
            self.hidden_state = next_hidden.detach()
            return actions


        if active_mask.shape != batch_shape:
            raise ValueError("active_mask must have shape [B,N]")
        mask = active_mask.to(
            device=observations.device, dtype=torch.bool
        ).unsqueeze(-1)


        if self.hidden_state is None:
            previous = self.initial_state(observations)
        self.hidden_state = torch.where(
            mask, next_hidden, previous
        ).detach()
        return actions * mask.to(dtype=actions.dtype)


    def reset_episode(self) -> None:
        """Clear all slot hidden state for a fresh episode."""
        self.hidden_state = None

