from abc import abstractmethod
from typing import List

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class TrainerPropsBase:
    @property
    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        pass

    @property
    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        pass

    @property
    def device(self) -> torch.device:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @property
    @abstractmethod
    def optimizer(self) -> Optimizer:
        pass

    @property
    @abstractmethod
    def criterion(self) -> Module:
        pass


class TrainerProps(TrainerPropsBase):
    def __init__(self,
                 datasets: List[Dataset],
                 device: torch.device,
                 optimizer: Optimizer,
                 criterion: Module,
                 train_shuffle=True):
        if not datasets:
            raise ValueError("データセットが指定されていません")

        self.datasets: List[Dataset] = datasets
        self.__device: torch.device = device
        self.__criterion: Module = criterion
        self.__optimizer: Optimizer = optimizer
        self.train_shuffle: bool = train_shuffle

    @property
    def train_dataloader(self) -> DataLoader:
        return DataLoader(ConcatDataset(self.datasets), shuffle=self.train_shuffle)

    @property
    def test_dataloader(self) -> DataLoader:
        return DataLoader(ConcatDataset(self.datasets), shuffle=False)

    @property
    def optimizer(self) -> Optimizer:
        return self.__optimizer

    @property
    def criterion(self) -> Module:
        return self.__criterion
