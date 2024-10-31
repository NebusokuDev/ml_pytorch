from torch import Tensor

from model.layer import Layer


class Encoder(Layer):
    def forward(self, *data: Tensor) -> Tensor:
        pass

class BottleNeck(Layer):

    def forward(self, *data: Tensor) -> Tensor:
        pass


class Decoder(Layer):
    def forward(self, *data: Tensor) -> Tensor:
        pass
