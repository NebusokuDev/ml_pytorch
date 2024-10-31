from typing import Optional

from torch import nn, Tensor, Module


class BottleNeck(Module):
    def __init__(self, in_channels: int, out_channels: int, disable: bool = False):
        super(BottleNeck, self).__init__()
        self.disable = disable

        # 畳み込み層とアクティベーションを定義
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.disable:
            return x

        # 畳み込み + ReLU の適用
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x

    def copy_with(self, disable: Optional[bool] = None):
        disable = disable if disable is not None else self.disable
        return BottleNeck(self.conv1.in_channels, self.conv2.out_channels, disable)
