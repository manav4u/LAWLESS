from abc import ABC, abstractmethod


from torch import Tensor
from torch.nn import Module




class Controller(Module, ABC):
    """Maps observations [B,N,23] to actions [B,N,2]."""


    observation_dim = 23
    action_dim = 2


    @abstractmethod
    def act(self, observations: Tensor, active_mask=None) -> Tensor:
        """Return bounded actions, zero for inactive slots."""

