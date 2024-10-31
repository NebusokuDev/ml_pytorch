import os
import zipfile
from logging import Logger, getLogger
from typing import Callable
from urllib.parse import urlparse
from zipfile import BadZipFile

import requests
from requests import Response
from tqdm import tqdm

KIB = 2 ^ 10


class Downloader:
    def __init__(self, root: str, url: str | Callable[[], str] = None, download: bool = True, zip_filename: str = None,
                 logger: Logger = None):
        self._root = root

        if isinstance(url, Callable):
            self.url = url()
        else:
            self.url = url

        self.zip_filename = zip_filename or os.path.basename(urlparse(self.url).path)  # URLからファイル名を取得
        self.zip_path = os.path.join(self._root, self.zip_filename)
        self.overwrite = download
        self.logger = logger or getLogger(__name__)

    def download(self):
        """データセットをダウンロードする"""
        if self.is_exist_zipfile() and not self.overwrite:
            print(f"Dataset already exists at {self.zip_path}, skipping download.")
            return

        print(f"Downloading dataset from {self.url}...")

        try:
            response = self.request(self.url)
            self.save_response_content(response, self.zip_path)

            print("\nDownload completed.")

        except requests.exceptions.RequestException as e:
            print(f"Error during download: {e}")

        except zipfile.BadZipFile:
            print("Error: Bad zip file.")

    def request(self, url):
        response = requests.get(url, stream=True)
        response.raise_for_status()  # HTTPエラーを確認
        return response

    @staticmethod
    def save_response_content(response: Response, destination, chunk_size: int = 100 * KIB):
        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=chunk_size)):
                if chunk:
                    f.write(chunk)

    def extract(self):
        """ZIPファイルを解凍する"""
        print(f"Unzipping {self.zip_path}...")
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                total_files = len(zip_ref.namelist())
                with tqdm(total=total_files, unit='file') as progress:
                    for file in zip_ref.namelist():
                        zip_ref.extract(file, self._root)
                        progress.update(1)
        except BadZipFile:
            print("Error: Bad zip file, extraction failed.")

        print(f"Extracted to {self._root}.")

    @property
    def root(self):
        return self._root

    def is_exist_zipfile(self):
        return os.path.exists(self.zip_path)

    def __call__(self, on_complete: Callable = None):
        """ダウンロードを実行する"""
        if self.overwrite or not self.is_exist_zipfile():
            self.download()
            self.extract()
        else:
            print("Dataset exists and 'overwrite' is False. No action taken.")

        if on_complete is not None:
            on_complete()


class GoogleDriveDownloader(Downloader):
    def __init__(self, root: str, file_id: str, download: bool = True, zip_filename=""):
        google_drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        self.file_id = file_id
        super().__init__(root, google_drive_url, download, zip_filename)

    def request(self, url: str):
        def get_confirm_token(response: Response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        session = requests.Session()

        response = session.get(url, params={'id': self.file_id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': self.file_id, 'confirm': token}
            return session.get(url, params=params, stream=True)

        return response
