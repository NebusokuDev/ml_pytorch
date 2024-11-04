from abc import abstractmethod, ABC
from datetime import datetime
from logging import getLogger, Logger
from os import path, makedirs
from typing import Optional

import torch
from pandas import Series, DataFrame
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchinfo import summary

CPU = "cpu"
CUDA = "cuda"

YY_MM_DD_FORMAT = "%Y_%m_%d--%H:%M"


class Trainer(ABC):
    """
    モデルのトレーニングおよび評価を行うための基本クラス。

    :param Logger logger: ロギング用のLoggerオブジェクト（指定がない場合はデフォルトのLoggerを使用）。
    """

    def __init__(self, logger: Optional[Logger] = None):
        """
        Trainerのコンストラクタ。

        :param Logger logger: ロギング用のLoggerオブジェクト（指定がない場合はデフォルトのLoggerを使用）。
        """
        self.logger = logger or getLogger(__name__)
        self._device: Optional[torch.device] = None
        self.__models: list[Module] = []

    def register_model(self, *models: Module):
        for model in models:
            self.__models.append(model)

    @abstractmethod
    def _setup_dataloader(self) -> tuple[DataLoader, DataLoader]:
        """
        データローダをセットアップするメソッド。

        :return:
            tuple[train_dataloader, test_dataloader]: トレーニングデータローダとテストデータローダ。
        """
        pass

    def _choose_device(self):
        """
        使用可能なデバイス（GPUまたはCPU）を選択するメソッド。
        """
        if torch.cuda.is_available():
            device = torch.device(CUDA)
            self.logger.info("Using GPU.")
        else:
            device = torch.device(CPU)
            self.logger.info("Using CPU.")
        return device

    def train(self, dataloader: DataLoader, batch_stride=100, device=None):
        """
        モデルをトレーニングするメソッド。

        :param DataLoader dataloader: トレーニングデータのDataLoader。
        :param int batch_stride: ログ出力の間隔（デフォルトは100）。
        :param device: モデルを配置するデバイス（指定がない場合は既存のデバイスを使用）。
        :return: トレーニング履歴のDataFrame。
        """
        history = []

        for model in self.__models:
            model.train()

        self._device = device or self._device

        self._transfer_model(self._device)

        for batch_index, (data, label) in enumerate(dataloader):
            try:
                transferred_data, transferred_label = self._transfer_data(data, label)
                snapshot = self._training_step(transferred_data, transferred_label)
                snapshot["batches"] = batch_index
                history.append(snapshot)

            except Exception as e:
                self.logger.exception(f"Error during training batch {batch_index}: {e}")
                raise e
            finally:
                self._release_device_cache(self._device)

            if batch_index % batch_stride == 0 and batch_index != 0:
                self._show_progress(batch_index, len(dataloader), snapshot)

        return DataFrame(history)

    def test(self, dataloader: DataLoader, batch_stride=100, device: torch.device = None):
        """
        モデルを評価するメソッド。

        :param DataLoader dataloader: テストデータのDataLoader。
        :param int batch_stride: ログ出力の間隔（デフォルトは100）。
        :param device: モデルを配置するデバイス（指定がない場合は既存のデバイスを使用）。
        :return: テスト履歴のDataFrame。
        """
        history = []

        for model in self.__models:
            model.eval()

        self._device = device or self._device
        self._transfer_model(self._device)

        with torch.no_grad():
            for batch_index, (data, label) in enumerate(dataloader):
                try:
                    transferred_data, transferred_label = self._transfer_data(data, label)
                    snapshot = self._test_step(transferred_data, transferred_label)
                    snapshot["batches"] = batch_index
                    history.append(snapshot)
                except Exception as e:
                    self.logger.error(f"Error {e}")
                    raise e
                finally:
                    self._release_device_cache(self._device)

                if batch_index % batch_stride == 0 and batch_index != 0:
                    self._show_progress(batch_index, len(dataloader), snapshot)

        return DataFrame(history)

    def _show_progress(self, batch_index, total, snapshot):
        progress = float(batch_index) / total * 100  # 進捗をパーセント表示
        self.logger.info(f"\t- [test #{batch_index}] success <progress: {progress:.2f}%> {snapshot}")

    @abstractmethod
    def _training_step(self, data: Tensor, label: Tensor) -> Series:
        """
        トレーニングステップを実行するための抽象メソッド。

        :param Tensor data: 入力データ。
        :param Tensor label: 対応するラベル。
        :return: 1エポックのスナップショットとしての結果のSeries。
        """
        pass

    @abstractmethod
    def _test_step(self, data: Tensor, label: Tensor) -> Series:
        """
        テストステップを実行するための抽象メソッド。

        :param Tensor data: 入力データ。
        :param Tensor label: 対応するラベル。
        :return: 1エポックのスナップショットとしての結果のSeries。
        """
        pass

    @staticmethod
    def _release_device_cache(device: torch.device):
        """
        デバイスのキャッシュを解放するメソッド。

        :param device: 解放するデバイス。
        """
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def _transfer_model(self, device: torch.device):
        """
        モデルを指定されたデバイスに転送するメソッド。

        :param device: 転送先のデバイス。
        """
        if self._device != device:
            for model in self.__models:
                model.to(device)
            self._device = device

    def _transfer_data(self, data, label) -> tuple[Tensor, Tensor]:
        return data.to(self._device), label.to(self._device)

    def _cleanup(self, device):
        """
        クリーンアップ処理を行うメソッド。

        :param device: クリーンアップするデバイス。
        """
        self._release_device_cache(device)

    def run(self, epoch: int, batch_stride=100, display_model_info=True) -> tuple[list[DataFrame], list[DataFrame]]:
        """
        学習プロセスを実行するメソッド。

        :param int epoch: 学習のエポック数。
        :param int batch_stride: ログ出力の間隔（デフォルトは100）。
        :param bool display_model_info: モデルの情報を表示するかどうか（デフォルトはTrue）。
        :return : トレーニングレポートとテストレポートのリスト。
        """
        train_report = []
        test_report = []

        (train_dataloader, test_dataloader) = self._setup_dataloader()
        self._device = self._choose_device()

        self.logger.info("--- leaning start! ---")

        if display_model_info:
            for model in self.__models:
                self.logger.info(f"\n{summary(model)}")

        self.logger.info("[setup]")
        self.logger.info("[setup success]")

        try:
            for epoch_count in range(epoch):
                self.logger.info(
                    f"[epoch: #{epoch_count + 1:03d}/{epoch:03d}, total: {(epoch_count + 1) / epoch:.2f}%]")
                self.logger.info(f"[training: #{epoch_count + 1}]")

                train_report.append(self.train(train_dataloader, batch_stride))

                self.logger.info(f"[test: #{epoch_count + 1}]")
                test_report.append(self.test(test_dataloader, batch_stride))

                self._release_device_cache(self._device)
        finally:
            self._cleanup(self._device)

        return train_report, test_report

    def save_model(self, model_name, root: str = "trained_model"):
        """
        モデルの状態をファイルに保存するメソッド。

        :param str root: 保存するデータのルートディレクトリ
        :param model_name: 保存するモデルの名前。
        """
        for index, model in enumerate(self.__models):
            date = datetime.now().strftime(YY_MM_DD_FORMAT)
            save_path = f"./{root}/{model_name}/{date}/{model_name}-{index}.pth"

            # ディレクトリが存在しない場合は作成
            makedirs(path.dirname(save_path), exist_ok=True)
            torch.save(model.to(CPU).state_dict(), save_path)
            self.logger.info(f"Model saved to {save_path}")

    def save_report(self, model_name: str, train_report: list[DataFrame], test_report: list[DataFrame],
                    root: str = "trained_model"):
        """
        学習およびテストのレポートをCSVファイルに保存するメソッド。

        :param str root: レポートを保存するルートディレクトリ
        :param model_name: レポートを保存するモデルの名前。
        :param train_report: トレーニングレポートのリスト。
        :param test_report: テストレポートのリスト。
        """
        date = datetime.now().strftime(YY_MM_DD_FORMAT)

        for i, epoch_report in enumerate(train_report):
            train_report_path = f"./{root}/{model_name}/{date}/{model_name}_epoch-{i}_train_report.csv"
            epoch_report.to_csv(train_report_path, index=False)  # CSVに保存
            self.logger.info(f"Train report saved to {train_report_path}")

        for i, epoch_report in enumerate(test_report):
            test_report_path = f"./{root}/{model_name}/{date}/{model_name}_epoch-{i}_test_report.csv"
            epoch_report.to_csv(test_report_path, index=False)  # CSVに保存
            self.logger.info(f"Test report saved to {test_report_path}")

    def build_model(self, model_name, epoch: int):
        """
        モデルを構築し、学習・評価結果を保存するメソッド。

        :param model_name: モデルの名前。
        :param epoch: 学習のエポック数。
        """
        train_report, test_report = self.run(epoch)
        self.save_report(model_name, train_report, test_report)
        self.save_model(model_name)

    def __call__(self, epoch: int, display_model_info=True, batch_stride=100) -> (list[DataFrame], list[DataFrame]):
        """
        Trainerオブジェクトを呼び出して学習プロセスを開始するメソッド。

        :param epoch: 学習のエポック数。
        :param display_model_info: モデルの情報を表示するかどうか（デフォルトはTrue）。
        :param batch_stride: ログ出力の間隔（デフォルトは100）。
        :return: トレーニングレポートとテストレポートのリスト。
        """
        return self.run(epoch, batch_stride, display_model_info)
