from copy import deepcopy
from typing import Optional
from torch import Tensor
from torch.nn import Module, MaxPool2d, Conv2d, ReLU


class Encoder(Module):
    def __init__(self, in_channels, out_channels, disable=False):
        super(Encoder, self).__init__()
        self.disable: bool = disable
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU(inplace=True)

    def forward(self, input_value: Tensor) -> Tensor:
        if self.disable:
            return input_value
        conv1_output = self.relu(self.conv1(input_value))
        conv2_output = self.relu(self.conv2(conv1_output))
        return self.pool(conv2_output)

    def copy_with(self, conv1: Optional[Module] = None,
                  conv2: Optional[Module] = None,
                  pool: Optional[Module] = None,
                  relu: Optional[Module] = None,
                  disable: Optional[bool] = None):
        conv1 = conv1 or deepcopy(self.conv1)
        conv2 = conv2 or deepcopy(self.conv2)
        pool = pool or deepcopy(self.pool)
        relu = relu or deepcopy(self.relu)

        # 明示的な None チェックを行い、None であれば self.disable を使用
        if disable is None:
            disable = self.disable

        instance = Encoder(self.in_channels, self.out_channels, disable)
        instance.conv1 = conv1
        instance.conv2 = conv2
        instance.pool = pool
        instance.relu = relu

        return instance