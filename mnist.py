from logging import getLogger
from typing import Any

from torch.nn import CrossEntropyLoss
from torchvision import transforms
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torchvision.datasets import MNIST

from choose_device import choose_device
from model.mnist.mnist import MnistCnn
from trainer.trainer_props import TrainerProps
from trainer.trariner import Trainer

logger = getLogger(__name__)

cnn = MnistCnn()


def setup() -> TrainerProps:
    transform = transforms.Compose([])

    mnist = MNIST(root="./assets")
    device = choose_device()
    criterion = CrossEntropyLoss()
    optimizer = Optimizer.Adam()
    return TrainerProps([mnist], device, optimizer, criterion)


def train(model, device, data, target, criterion, optimizer: Optimizer) -> dict[str, Any]:
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target).item()
    optimizer.step()
    return {"loss": loss}


def test(model, device, data, target, criterion) -> dict[str, Any]:
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)


report = Trainer(trainee_model=cnn, setup=setup, train_scenario=train, test_scenario=test, logger=logger).learn(20)
