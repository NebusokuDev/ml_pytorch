from torch import Tensor

from model.model import Model


class UNetLite(Model):
    def forward(self, input_data: Tensor) -> Tensor:
        pass

    def encode(self, input_data) -> list[Tensor]:
        pass

    def decode(self, input_data) -> list[Tensor]:
        pass
