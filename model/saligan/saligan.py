from torch import Tensor

from model.model import Model


class SaliGan(Model):
    def __init__(self, generator: Model, discriminator: Model):
        super(SaliGan, self).__init__()
        self.generator: Model = generator
        self.discriminator: Model = discriminator

    def forward(self, input_data: Tensor) -> Tensor:
        pass
