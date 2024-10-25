from torch import relu
from torch.nn import Linear

from model.model import Model


class MnistCnn(Model):

    def __init__(self):
        super(MnistCnn, self).__init__()
        self.fc1 = Linear(28 * 28, 128)  # 入力層
        self.fc2 = Linear(128, 10)  # 出力層

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # 28x28の画像を1次元に変換
        x = relu(self.fc1(x))
        x = self.fc2(x)
        return x
