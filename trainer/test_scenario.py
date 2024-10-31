from abc import ABCMeta, abstractmethod

from torch import Tensor
from torch.nn import Module

from model.model import Model


class TestScenario(metaclass=ABCMeta):
    @abstractmethod
    def test(self,
             model: Model,
             device,
             input_data: Tensor,
             target_data: Tensor,
             criterion: Module
             ) -> dict[str, any]:
        pass

    def __call__(self, model: Model,
                 device,
                 input_data: Tensor,
                 target_data: Tensor,
                 criterion: Module):
        return self.test(model, device, input_data, target_data, criterion)
