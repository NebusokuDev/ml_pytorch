from torch import Tensor

from model.model import Model
from model.saligan.layers import Generator, Discriminator


class SaliGan(Model):
    def __init__(self, generator: Model = None, discriminator: Model = None):
        super(SaliGan, self).__init__()
        self.generator: Model = generator or Generator()
        self.discriminator: Model = discriminator or Discriminator()

    def forward(self, input_data: Tensor) -> tuple[Tensor, Tensor]:
        generated_data = self.generate(input_data)
        validity = self.discriminator(input_data)
        return validity, generated_data

    def generate(self, input_data: Tensor) -> Tensor:
        return self.generator(input_data)

    def discriminate(self, input_data: Tensor) -> Tensor:
        return self.discriminator(input_data)
