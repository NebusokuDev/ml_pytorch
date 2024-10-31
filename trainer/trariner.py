import json
from abc import ABCMeta, abstractmethod
from datetime import datetime
from logging import getLogger, Logger
from typing import Any, Callable, Optional

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

    def __init__(self, logger: Optional[Logger], trainee_model: Model):
        self.trainee_model = trainee_model
        self.logger = logger or getLogger(__name__)
        self._trainee_model_device = ""

    @abstractmethod
    def _setup(self) -> dict | tuple:
        pass

    def train(self, dataloader: DataLoader, criterion: Module, optimizer: Optimizer, device, batch_stride=100):
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

            if batch_index % batch_stride == 0 and batch_index != 0:
                progress = float(batch_index) / len(dataloader) * 100  # 進捗をパーセント表示

                self.logger.info(
                    f"\t- [training #{batch_index}] success <progress: {progress:.2f}%> {snapshot}")

        return history

    def test(self, dataloader: DataLoader, criterion: Module, device, batch_stride=100):
        history = []
        self.trainee_model.eval()
        self._transfer_model(device)

        with torch.no_grad():
            for batch_index, (input_data, target_data) in enumerate(dataloader):
                try:
                    snapshot = self._test_scenario(self.trainee_model, device, input_data, target_data, criterion)
                    history.extend(snapshot)
                except Exception as e:
                    self.logger.error(f"Error {e}")

                    raise e

                if batch_index % batch_stride == 0 and batch_index != 0:
                    progress = float(batch_index) / len(dataloader) * 100  # 進捗をパーセント表示
                    self.logger.info(
                        f"\t- [test #{batch_index}] success <progress: {progress:.2f}%> {snapshot}")

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

    def learn(self, epoch: int, display_model_info=True, batch_stride=100) -> (dict[str, Any], dict[str, Any]):
        train_report = []
        test_report = []

        self.logger.info("--- leaning start! ---")

        if display_model_info:
            self.logger.info(f"\n{summary(self.trainee_model)}")

        self.logger.info("[setup]")
        props = self._setup()
        self.logger.info("[setup success]")
        self.logger.info(f"\t- device: {props.device}")
        self.logger.info(f"\t- criterion: {props.criterion}")
        self.logger.info(f"\t- optimizer: {type(props.optimizer)}")
        self.logger.info(f"\t- criterion: {type(props.criterion)}")
        self.logger.info(f"\t- train_dataloader: <batch size: {props.train_dataloader.batch_size}>")
        self.logger.info(f"\t- test_dataloader: <batch size: {props.test_dataloader.batch_size}>")

        try:
            for epoch_count in range(epoch):
                self.logger.info(
                    f"[epoch: #{epoch_count + 1:03d}/{epoch:03d}, total: {(epoch_count + 1) / epoch:.2f}%]")
                self.logger.info(f"[training: #{epoch_count + 1}]")
                self.train(props.train_dataloader, props.criterion, props.optimizer, props.device, batch_stride)
                self.logger.info(f"[test: #{epoch_count + 1}]")
                self.test(props.test_dataloader, props.criterion, props.device, batch_stride)

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
        train_report_path = f"./trained_models/{model_name}/{date}-{model_name}_train_report.json"
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
                 training_scenario:
                 Callable[[Model, torch.device, Tensor, Tensor, Module, Optimizer], dict[str, Any]]
                 | TrainingScenario,
                 test_scenario:
                 Callable[[Model, torch.device, Tensor, Tensor, Module], dict[str, Any]]
                 | TestScenario,
                 logger: Logger = None):
        super().__init__(trainee_model=trainee_model, logger=logger, )
        self.__training_scenario = training_scenario
        self.__test_scenario = test_scenario
        self.__setup_method = setup


def redefine_setup(self, setup_method):
    self.__setup_method = setup_method


def redefine_test_scenario(self, test_scenario):
    self.__test_scenario = test_scenario


def redefine_train_scenario(self, train_scenario):
    self.__training_scenario = train_scenario


def _setup(self) -> dict:
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
