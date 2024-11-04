from torch.nn import Module, Sequential, Conv2d, LeakyReLU, Upsample, Sigmoid, MaxPool2d, Tanh, Linear
from torchvision.models import vgg16


class Generator(Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder1 = vgg16(pretrained=pretrained).features[:17]
        self.encoder_last = vgg16(pretrained=pretrained).features[17:-1]
        self.decoder = Sequential(
            Conv2d(512, 512, 3, padding=1),
            LeakyReLU(),

            Conv2d(512, 512, 3, padding=1),
            LeakyReLU(),

            Conv2d(512, 512, 3, padding=1),
            LeakyReLU(),

            Upsample(scale_factor=2),

            Conv2d(512, 512, 3, padding=1),
            LeakyReLU(),

            Conv2d(512, 512, 3, padding=1),
            LeakyReLU(),

            Conv2d(512, 512, 3, padding=1),
            LeakyReLU(),

            Upsample(scale_factor=2),

            Conv2d(512, 256, 3, padding=1),
            LeakyReLU(),

            Conv2d(256, 256, 3, padding=1),
            LeakyReLU(),

            Conv2d(256, 256, 3, padding=1),
            LeakyReLU(),

            Upsample(scale_factor=2),

            Conv2d(256, 128, 3, padding=1),
            LeakyReLU(),

            Conv2d(128, 128, 3, padding=1),
            LeakyReLU(),

            Upsample(scale_factor=2),

            Conv2d(128, 64, 3, padding=1),
            LeakyReLU(),

            Conv2d(64, 64, 3, padding=1),
            LeakyReLU(),

            Conv2d(64, 1, 1, padding=0),
            Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder_last(x)
        x = self.decoder(x)
        return x

class Discriminator(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = Sequential(
            Conv2d(4, 3, 1, padding=1),
            LeakyReLU(inplace=True),
            Conv2d(3, 32, 3, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2, stride=2),
            Conv2d(32, 64, 3, padding=1),
            LeakyReLU(inplace=True),
            Conv2d(64, 64, 3, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2, stride=2),
            Conv2d(64, 64, 3, padding=1),
            LeakyReLU(inplace=True),
            Conv2d(64, 64, 3, padding=1),
            LeakyReLU(inplace=True),
            MaxPool2d(2, stride=2))
        self.classifier = Sequential(
            Linear(64 * 32 * 24, 100, bias=True),
            Tanh(),
            Linear(100, 2, bias=True),
            Tanh(),
            Linear(2, 1, bias=True),
            Sigmoid())

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)

        return x