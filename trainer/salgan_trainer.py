from pandas import Series
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from trainer.trariner import Trainer


class SalGANTrainer(Trainer):
    def _setup_dataloader(self) -> (DataLoader, DataLoader):
        pass

    def _training_step(self, model: Module, data: Tensor, label: Tensor) -> Series:
        pass

    def _test_step(self, model: Module, data: Tensor, label: Tensor) -> Series:
        pass
