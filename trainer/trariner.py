import json
from abc import ABCMeta, abstractmethod
from datetime import datetime
from logging import getLogger, Logger
from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchinfo import summary

from model.model import Model
from trainer.setup import Setup
from trainer.test_scenario import TestScenario
from trainer.trainer_props import TrainerPropsBase
from trainer.training_senario import TrainingScenario

CPU = "cpu"

YY_MM_DD_FORMAT = "%Y_%m_%d--%hh:%mm"


class TrainerBase(metaclass=ABCMeta):

    def __init__(self, logger, trainee_model: Model):
        self.trainee_model = trainee_model
        self.logger = logger or getLogger(__name__)
        self._trainee_model_device = ""

    @abstractmethod
    def _setup(self) -> TrainerPropsBase:
        pass

    def train(self, dataloader: DataLoader, criterion: Module, optimizer: Optimizer, device):
        history = []

        self.trainee_model.train()

        self._transfer_model(device)

        for batch_index, (input_data, target_data) in enumerate(dataloader):
            try:
                snapshot = self._training_scenario(self.trainee_model,
                                                   device,
                                                   input_data,
                                                   target_data,
                                                   criterion,
                                                   optimizer)
                history.extend(snapshot)

            except Exception as e:
                self.logger.error(f"Error during training batch {batch_index}: {e}")

                raise e

            self.logger.info(f"<batch: #{batch_index}> success!")

        return history

    def test(self, dataloader: DataLoader, criterion: Module, device):
        history = []
        self.trainee_model.eval()
        self._transfer_model(device)

        with torch.no_grad():
            for batch_index, (data, target) in enumerate(dataloader):
                try:
                    snapshot = self._test_scenario(self.trainee_model, device, data, target, criterion)
                    history.extend(snapshot)
                except Exception as e:
                    self.logger.error(f"Error {e}")

                    raise e

                self.logger.info(f"<batch: #{batch_index}> success!")

        return history

    @abstractmethod
    def _training_scenario(self,
                           model: Model,
                           device: torch.device,
                           input_data: Tensor,
                           target_data: Tensor,
                           criterion: Module,
                           optimizer: Optimizer
                           ) -> dict[str, Any]:
        pass

    @abstractmethod
    def _test_scenario(self,
                       model: Model,
                       device: torch.device,
                       input_data: Tensor,
                       target_data: Tensor,
                       criterion: Module) -> dict[str, Any]:
        pass

    @staticmethod
    def _release_device_cache(device):
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _transfer_model(self, device):
        if self._trainee_model_device != device:
            self._trainee_model_device = self.trainee_model.to(device)

    def _cleanup(self, device):
        self._release_device_cache(device)

    def learn(self, epoch: int, display_model_info=True) -> (dict[str, Any], dict[str, Any]):
        train_report = []
        test_report = []

        self.logger.info("---leaning start!---")

        if display_model_info:
            self.logger.info(summary(self.trainee_model))

        self.logger.info("---- setup start! ----")
        props = self._setup()
        self.logger.info("---- setup success!----")

        try:
            for epoch_count in range(epoch):
                self.logger.info(f"[epoch: #{epoch_count + 1:3d}/{epoch}, {(epoch_count + 1) / epoch:.2f}%]")
                self.logger.info("---- train start! ----")
                self.train(props.train_dataloader, props.criterion, props.optimizer, props.device)
                self.logger.info("---- train success! ----")
                self.logger.info("---- test start! ----")
                self.test(props.test_dataloader, props.criterion, props.device)
                self.logger.info("---- test success! ----")

                self._release_device_cache(props.device)
        finally:
            self._cleanup(props.device)

        return train_report, test_report

    def save_model(self, model_name):
        date = datetime.now().strftime(YY_MM_DD_FORMAT)

        save_path = f"./models/{model_name}/{date}-{model_name}.pth"
        torch.save(self.trainee_model.to(CPU).state_dict(), save_path)
        self.logger.info(f"Model saved to {save_path}")

    def save_report(self, model_name: str, train_report: dict[str, Any], test_report: dict[str, Any]):
        date = datetime.now().strftime(YY_MM_DD_FORMAT)
        train_report_path = f"./models/{model_name}/{date}-{model_name}_train_report.json"
        test_report_path = f"./models/{model_name}/{date}-{model_name}_test_report.json"

        # Save reports to JSON
        with open(train_report_path, 'w') as f:
            json.dump(train_report, f)

        with open(test_report_path, 'w') as f:
            json.dump(test_report, f)

        self.logger.info(f"Reports saved to {train_report_path} and {test_report_path}")

    def build_model(self, model_name, epoch: int):
        train_report, test_report = self.learn(epoch)
        self.save_report(model_name, train_report, test_report)
        self.save_model(model_name)


class Trainer(TrainerBase):
    def __init__(self,
                 trainee_model: Model,
                 setup: Callable[[], TrainerPropsBase] | Setup,
                 train_scenario:
                 Callable[[Model, torch.device, Tensor, Tensor, Module, Optimizer], dict[str, Any]]
                 | TrainingScenario,
                 test_scenario:
                 Callable[[Model, torch.device, Tensor, Tensor, Module], dict[str, Any]]
                 | TestScenario,
                 logger: Logger = None):
        super().__init__(logger, trainee_model)

        if train_scenario is TrainingScenario:
            self.__training_scenario = train_scenario.training
        else:
            self.__training_scenario = train_scenario

        if test_scenario is TestScenario:
            self.__test_scenario = test_scenario.test
        else:
            self.__test_scenario = test_scenario

        if setup is Setup:
            self.__setup_method = setup.setup
        else:
            self.__setup_method = setup

    def redefine_setup(self, setup_method):
        self.__setup_method = setup_method

    def redefine_test_scenario(self, test_scenario):
        self.__test_scenario = test_scenario

    def redefine_train_scenario(self, train_scenario):
        self.__training_scenario = train_scenario

    def _setup(self) -> TrainerPropsBase:
        return self.__setup_method()

    def _training_scenario(self,
                           model: Model,
                           device: torch.device,
                           input_data: Tensor,
                           target_data: Tensor,
                           criterion: Module,
                           optimizer: Optimizer
                           ) -> dict[str, any]:
        return self.__training_scenario(model, device, input_data, target_data, criterion, optimizer)

    def _test_scenario(self,
                       model: Model,
                       device,
                       input_data: Tensor,
                       target_data: Tensor,
                       criterion: Module
                       ) -> dict[str, any]:
        return self.__test_scenario(model, device, input_data, target_data, criterion, )
