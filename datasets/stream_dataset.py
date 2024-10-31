from abc import abstractmethod, ABC
from os import PathLike
from typing import Type, Optional

from torch import Tensor
from torch.utils.data import Dataset

from datasets.downloader import Downloader

"""
python ver. 3.12
"""


class StreamDataset[TData, TTarget](Dataset, ABC):
    def __init__(self, url: str, downloader: Optional[Downloader] = None, root: str | PathLike = "./data",
                 download: bool = False):
        super().__init__()
        self._root: str = str(root) if isinstance(root, PathLike) else root

        self._downloader = downloader or Downloader(url, self._root, overwrite=download)

    def __getitem__(self, item):
        raw_data, raw_target = self._get_raw_data_pair(item)

        data = self._convert_data_to_tensor(raw_data)
        target = self._convert_target_to_tensor(raw_target)

        return data, target

    def __iter__(self):
        for index in range(len(self)):
            yield self.__getitem__(index)

    def __len__(self) -> int:
        return self._get_length()

    @abstractmethod
    def download(self):
        self._downloader.download()
        self._downloader._root()

    @abstractmethod
    def _convert_data_to_tensor(self, data) -> Tensor:
        pass

    @abstractmethod
    def _convert_target_to_tensor(self, target) -> Tensor:
        pass

    @abstractmethod
    def _get_raw_data_pair(self, index) -> tuple[TData, TTarget]:
        pass

    @abstractmethod
    def _get_length(self) -> int:
        pass

    @abstractmethod
    def _data_type(self) -> Type:
        pass

    @abstractmethod
    def _target_type(self) -> Type:
        pass
