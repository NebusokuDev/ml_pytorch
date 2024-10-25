from torch import Tensor

from model.model import Model


class Discriminator(Model):
    def forward(self, input_data: Tensor) -> Tensor:
        pass
