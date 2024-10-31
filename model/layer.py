from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class Layer(Module, ABC):
    @abstractmethod
    def forward(self, data: Tensor):
        pass
