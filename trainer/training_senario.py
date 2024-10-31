from abc import ABCMeta, abstractmethod
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

from model.model import Model


class TrainingScenario(metaclass=ABCMeta):
    """
    1バッチのトレーニングシナリオを定義する抽象クラス。
    セクションを複数に分けたいトレーニングシナリオに対応します。

    Examples:
        ```python: MyTrainingScenario.py
        class MyTrainingScenario(TrainingScenario):
            def training(self, model, device, input_data, target_data, criterion, optimizer):
                self.training_generator(model, device, input_data, target_data, criterion, optimizer)
                self.training_discriminator(model, device, input_data, target_data, criterion, optimizer)

            def training_generator(self, model, device, input_data, target_data, criterion, optimizer):
                # ジェネレーターのトレーニング内容
                pass

            def training_discriminator(self, model, device, input_data, target_data, criterion, optimizer):
                # 判別器のトレーニング内容
                pass
        ```
    """

    @abstractmethod
    def training(self,
                 model: Model,
                 device: torch.device,
                 input_data: Tensor,
                 target_data: Tensor,
                 criterion: Module,
                 optimizer: Optimizer
                 ) -> dict[str, Any]:
        """
            1バッチのトレーニングシナリオを定義する抽象メソッド。

            Args:
                - model (Model): 学習させるモデル。
                - device (torch.device): モデルとデータが配置されるデバイス。
                - input_data (Tensor): 入力データ。
                - target_data (Tensor): 対象データ。
                - criterion (Module): 損失関数。
                - optimizer (Optimizer): 最適化アルゴリズム。

            Returns:
                dict[str, Any]: 1バッチあたりのトレーニング結果を含む辞書。
            """
        pass

    def __call__(self,
                 model: Model,
                 device: torch.device,
                 input_data: Tensor,
                 target_data: Tensor,
                 criterion: Module,
                 optimizer: Optimizer
                 ):
        return self.training(model, device, input_data, target_data, criterion, optimizer)


class Compose(TrainingScenario):
    def __init__(self, *training_scenario: [TrainingScenario]):
        self.__training_scenario = training_scenario

    def training(self, model: Model, device: torch.device, input_data: Tensor, target_data: Tensor, criterion: Module,
                 optimizer: Optimizer) -> dict[str, Any]:
        results = []
        for section_index, section, in enumerate(self.__training_scenario):
            snapshot = section.training(model, device, input_data, target_data, criterion, optimizer)
            results.append(snapshot)

        return results
