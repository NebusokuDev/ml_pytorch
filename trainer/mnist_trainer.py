from pandas import Series
from torch import Tensor
from torch.utils.data import DataLoader

from trainer.trariner import Trainer


class MNISTTrainer(Trainer):
    def _setup_dataloader(self) -> tuple[DataLoader, DataLoader]:
        pass

    def _training_step(self, data: Tensor, label: Tensor) -> Series:
        pass

    def _test_step(self, data: Tensor, label: Tensor) -> Series:
        pass