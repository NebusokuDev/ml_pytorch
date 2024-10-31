from torch import Tensor

from model.model import Model


class UNetLite(Model):
    def __init__(self, encoder, decoder, ):
        super(UNetLite, self).__init__()
    def forward(self, input_data: Tensor) -> Tensor:
        pass
