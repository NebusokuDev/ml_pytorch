from torch import Tensor
from torch.ao.nn.quantized import BatchNorm2d
from torch.nn import Conv2d, Sequential, ReLU, Sigmoid, LeakyReLU

from model.model import Model


class Generator(Model):
    # rgb(3) -> grayscale(1)
    def __init__(self, input_color_ch=3, output_color_ch=1):
        super().__init__()

        self.layer1 = Sequential(
            Conv2d(input_color_ch, 64, kernel_size=4, stride=2, padding=1),
            ReLU(True)
        )

        self.layer2 = Sequential(
            Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(128),
            ReLU(True)
        )

        self.layer3 = Sequential(
            Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(256),
            ReLU(True)
        )

        self.layer4 = Conv2d(256, output_color_ch, kernel_size=4, stride=2, padding=1)

        self.output = Sigmoid()

    def forward(self, input_data: Tensor) -> Tensor:
        l1 = self.layer1(input_data)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        return self.output(l4)


# 判別ネットワーク
class Discriminator(Model):
    # rgb + grayscale -> bool(0-1)
    def __init__(self, input_color_ch=3 + 1):
        super(Discriminator, self).__init__()
        self.model = Sequential(
            Conv2d(input_color_ch, 64, kernel_size=4, stride=2, padding=1),
            LeakyReLU(0.2, inplace=True),
            Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            BatchNorm2d(128),
            LeakyReLU(0.2, inplace=True),
            Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
            Sigmoid()  # 0-1の確率を出力
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)
