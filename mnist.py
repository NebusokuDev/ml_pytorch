from logging import getLogger, config
from typing import Any

from torch import optim
from torch.nn import CrossEntropyLoss
from torch.optim import Optimizer
from torchvision import transforms
from torchvision.datasets import MNIST
from yaml import safe_load

from util.choose_device import choose_device
from model.mnist.mnist import MnistCnn
from trainer.trainer_props import TrainerProps
from trainer.trariner import Trainer

with open("./logger.config.yaml", "r") as yml:
    logger_config = safe_load(yml)

config.dictConfig(logger_config)

cnn = MnistCnn()


def setup():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist = MNIST(root="./data", download=True, transform=transform)
    device = choose_device()
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters())
    return


def train(model, device, data, target, criterion, optimizer: Optimizer) -> dict[str, Any]:
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target).item()
    optimizer.step()
    return {"loss": loss}


def test(model, device, data, target, criterion) -> dict[str, Any]:
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target).item()
    return {"loss": loss}


logger = getLogger()

trainer = Trainer(trainee_model=cnn, setup=setup, training_scenario=train, test_scenario=test, logger=logger)

trainer.learn(epoch=20, batch_stride=1000)
