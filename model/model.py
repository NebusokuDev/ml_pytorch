from abc import abstractmethod, ABC

from torch import Tensor
from torch.nn import Module


class Model(Module, ABC):
    @abstractmethod
    def forward(self, input_data: Tensor):
        """
        モデルの順伝播を定義します。

        Args:
            input_data (Tensor): モデルへの入力データ。

        Returns:
            Tensor: モデルからの出力データ。
        """
        pass
