import zipfile
from logging import Logger, getLogger
from os import path

import requests
from tqdm import tqdm


class Downloader:
    def __init__(self, url: str, root: str = "", overwrite: bool = True, logger: Logger = None):
        self._root = root
        self._overwrite = overwrite
        self._url = url
        self._logger = logger or getLogger(__name__)
        self._extracted_files = []  # 解凍されたファイル名を保持するリスト

    @property
    def zip_path(self):
        return str(path.join(self._root, self.get_filename_from_url()))

    def is_downloaded(self) -> bool:
        """データセットがダウンロードされているかを確認"""
        return path.exists(self.zip_path)

    def file_extracted(self) -> bool:
        """ファイルが解凍されているかを確認"""
        return bool(self._extracted_files)  # リストが空でないか確認

    def download(self) -> None:
        """データセットをダウンロードする"""
        if self.is_downloaded() and not self._overwrite:
            self._logger.info(f"Dataset already exists at {self.zip_path}, skipping download.")
            return

        self._logger.info(f"Downloading dataset from {self._url}...")

        try:
            response = requests.get(self._url, stream=True)
            response.raise_for_status()  # HTTPエラーを確認
            total_size = int(response.headers.get('content-length', 0))

            with tqdm(total=total_size, unit='B', unit_scale=True, dynamic_ncols=True) as progress:
                with open(self.zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=128):
                        f.write(chunk)
                        progress.update(len(chunk))

            self._logger.info("\nDownload completed.")

        except requests.exceptions.RequestException as err:
            self._logger.error(f"Error during download: {err}")
            
    def extract(self) -> None:
        """ZIPファイルを解凍する"""
        if self.file_extracted() and not self._overwrite:
            self._logger.info("Files already extracted, skipping extraction.")
            return

        if not self.is_downloaded():
            self._logger.error(f"ZIP file not found: {self.zip_path}")
            return

        self._logger.info(f"Unzipping {self.zip_path}...")
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(self._root)
                self._extracted_files = zip_ref.namelist()  # 解凍したファイル名をリストに保存

            self._logger.info(f"Extracted to {self._root}.")

        except zipfile.BadZipFile:
            self._logger.error("Error: Bad zip file.")
        except Exception as e:
            self._logger.error(f"Error during extraction: {e}")

    def get_filename_from_url(self) -> str:
        """URLからファイル名を取得する"""
        return self._url.split("/")[-1]

    def get_extracted_files(self):
        """解凍されたファイル名を取得する"""
        return self._extracted_files

    def __call__(self) -> None:
        """オブジェクトを呼び出すときにダウンロードと解凍を実行する"""
        self.download()
        self.extract()
