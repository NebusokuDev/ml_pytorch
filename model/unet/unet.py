from torch.nn import Module

from model.model import Model


class UNet(Model):
    def __init__(self):
        super(UNet, self).__init__()

    def forward(self, input_data):
        return input_data
