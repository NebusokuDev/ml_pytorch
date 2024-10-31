from typing import Optional
from torch import Tensor
from torch.nn import Module, ConvTranspose2d, ReLU


class Decoder(Module):
    def __init__(self, in_channels: int, out_channels: int, activate: Optional[Module] = None, disable=False):
        super(Decoder, self).__init__()
        self.disable: bool = disable

        self.deconv = ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.activate = ReLU(inplace=True)

    def forward(self, input_value: Tensor) -> Tensor:
        if self.disable:
            return input_value

        return self.activate(self.deconv(input_value))

    def copy_with(self, activate: Optional[Module] = None, disable: Optional[bool] = None):
        disable = disable if disable is not None else self.disable
        activate = activate or self.activate
        return Decoder(self.deconv.in_channels, self.deconv.out_channels, activate, disable)
