import os
import requests
from os import path
from logging import Logger, getLogger
from typing import Optional
from tqdm import tqdm
import zipfile


class Downloader:
    """
    データセットをダウンロードするためのハンドラ
    """

    def __init__(self, url: str, download_path: str = "./assets/", overwrite: bool = False,
                 logger: Optional[Logger] = None):
        self.logger = logger or getLogger(__name__)
        self.url = url
        self.overwrite = overwrite
        self.download_path = download_path
        self.downloaded_file_name = ""

    def download(self):
        if self.is_downloaded() and not self.overwrite:
            self.logger.info(f"{self.download_path} はすでに存在します。")
            return

        self.logger.info(f"ダウンロードを開始します: {self.url}")

        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()  # ステータスコードが200以外の場合、例外を発生させる

            total_size = int(response.headers.get('content-length', 0))
            with open(self.download_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=self.download_path) as pbar:
                    for data in response.iter_content(chunk_size=4096):
                        f.write(data)
                        pbar.update(len(data))  # ダウンロードしたバイト数を更新
            self.logger.info(f"{self.download_path} にダウンロード完了。")
        except requests.HTTPError as e:
            self.logger.error(f"HTTPエラーが発生しました: {e}")
        except requests.RequestException as e:
            self.logger.error(f"ダウンロード中にエラーが発生しました: {e}")

    def unzip(self):
        if not path.exists(self.download_path):
            self.logger.error(f"{self.download_path} は存在しません。解凍できません。")
            return

        extract_path = path.dirname(self.download_path)  # 解凍先を指定
        try:
            with zipfile.ZipFile(self.download_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)  # 同じディレクトリに解凍
            self.logger.info(f"{self.download_path} の解凍が完了しました。")
        except zipfile.BadZipFile:
            self.logger.error(f"{self.download_path} は無効なZIPファイルです。")
        except Exception as e:
            self.logger.error(f"解凍中にエラーが発生しました: {e}")

    def exec(self):
        self.download()
        self.unzip()

    def downloaded_path(self) -> str:
        """ 解凍されたフォルダのパスを取得して返す """
        rel_path= path.join(self.download_path, self.downloaded_file_name)

        return path.dirname(lel)  # 解凍先のディレクトリパスを返す

    def is_downloaded(self) -> bool:
        result = path.exists(self.download_path)
        self.logger.info(f"{self.download_path} の存在確認: {result}")
        return result
