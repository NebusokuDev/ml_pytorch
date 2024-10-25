from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from model.model import Model


class TrainingScenario(metaclass=ABCMeta):

    @abstractmethod
    def training(self,
                 model: Model,
                 device: torch.device,
                 input_data: Tensor,
                 target_data: Tensor,
                 criterion: Module,
                 optimizer: Optimizer
                 ) -> dict[str, any]:
        pass
