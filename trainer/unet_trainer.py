from typing import Tuple

import torch
from pandas import Series
from torch import nn, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Compose, ToTensor, Resize

from datasets.cat2000 import Cat2000
from trainer.trariner import Trainer
import torch_directml

class UNetTrainer(Trainer):
    """
    U-Netモデルのトレーニングおよび評価を行うクラス。
    """

    def __init__(self, trainee_model: Module):
        super().__init__()
        self.trainee_model = trainee_model
        self.optimizer = torch.optim.Adam(self.trainee_model.parameters())  # 最初のモデルがU-Netであると仮定
        self.criterion = nn.BCEWithLogitsLoss()  # 適切な損失関数を使用
        self.register_model(self.trainee_model)

    # def _choose_device(self):
    #     try:
    #
    #         return torch_directml.device()
    #     except ImportError:
    #         return super()._choose_device()

    def _setup_dataloader(self) -> Tuple[DataLoader, DataLoader]:
        height = 1920 // 4
        width = 1080 // 4

        image_transform = Compose([Resize((width, height)), ToTensor()])
        map_transform = Compose([Resize((width, height)), ToTensor()])

        cat2000_train = Cat2000(image_transform=image_transform, map_transform=map_transform)
        cat2000_test = Cat2000(image_transform=image_transform, map_transform=map_transform)

        train_dataloader = DataLoader(ConcatDataset([cat2000_train]), shuffle=True)
        test_dataloader = DataLoader(ConcatDataset([cat2000_test]), shuffle=False)
        return train_dataloader, test_dataloader

    def _training_step(self, data: Tensor, label: Tensor) -> Series:
        """
        トレーニングステップを実行します。

        :param Tensor data: 入力データ。
        :param Tensor label: 対応するラベル。
        :return: トレーニング結果のSeries。
        """

        self.optimizer.zero_grad()
        outputs = self.trainee_model(data)
        loss = self.criterion(outputs, label)
        loss.backward()
        self.optimizer.step()

        return Series({'loss': loss.item()})

    def _test_step(self, data: Tensor, label: Tensor) -> Series:
        """
        テストステップを実行します。

        :param Tensor data: 入力データ。
        :param Tensor label: 対応するラベル。
        :return: テスト結果のSeries。
        """
        outputs = self.trainee_model(data)
        loss = self.criterion(outputs, label)

        # サリエンシーマップの生成
        saliency_map = self._generate_saliency_map(data, outputs)

        return Series({'loss': loss.item(), 'saliency_map': saliency_map})

    def _generate_saliency_map(self, input_data: Tensor, output: Tensor) -> Tensor:
        """
        サリエンシーマップを生成します。

        :param Tensor input_data: 入力データ。
        :param Tensor output: モデルの出力。
        :return: サリエンシーマップ。
        """
        input_data.requires_grad_()
        output = self.trainee_model(input_data)  # フォワードパス
        output_idx = output.argmax(dim=1)  # 出力がセグメンテーションマップであると仮定
        one_hot_output = torch.zeros_like(output).scatter_(1, output_idx.unsqueeze(1), 1)
        output_sum = output * one_hot_output  # 関連する出力のみを保持
        output_sum.backward()  # 勾配を取得するために逆伝播

        saliency_map = input_data.grad.data.abs()
        saliency_map = saliency_map / saliency_map.max()  # 正規化
        return saliency_map
