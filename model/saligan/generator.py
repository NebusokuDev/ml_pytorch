from torch import Tensor

from model.model import Model


class Generator(Model):
    def forward(self, input_data: Tensor) -> Tensor:
        pass