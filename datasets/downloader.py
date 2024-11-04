import os
import zipfile
from logging import Logger, getLogger
from os import mkdir
from os.path import exists
from typing import Callable
from urllib.parse import urlparse
from zipfile import BadZipFile

import requests
from requests import Response
from tqdm import tqdm

KIB = 2 ** 10


class Downloader:
    """
    指定されたURLからデータセットをダウンロードし、ZIPファイルを解凍するクラス。

    :param root: ダウンロード先のルートディレクトリ。
    :param url: ダウンロードするURLまたはURLを返す関数。
    :param overwrite: 既存のファイルを上書きするかどうか（デフォルトはFalse）。
    :param zip_filename: ZIPファイルの名前（指定がない場合はURLから取得）。
    :param logger: ロギング用のLoggerオブジェクト（指定がない場合はデフォルトのLoggerを使用）。
    """

    def __init__(self, root: str, url: str | Callable[[], str] = None, overwrite: bool = False,
                 zip_filename: str = None,
                 logger: Logger = None):
        """
        Downloaderのコンストラクタ。

        :param root: ダウンロード先のルートディレクトリ。
        :param url: ダウンロードするURLまたはURLを返す関数。
        :param overwrite: 既存のファイルを上書きするかどうか（デフォルトはFalse）。
        :param zip_filename: ZIPファイルの名前（指定がない場合はURLから取得）。
        :param logger: ロギング用のLoggerオブジェクト（指定がない場合はデフォルトのLoggerを使用）。
        """
        self._root = os.path.normpath(root)

        if isinstance(url, Callable):
            self.url = url()
        else:
            self.url = url

        self.zip_filename = zip_filename or os.path.basename(urlparse(self.url).path)  # URLからファイル名を取得
        self.zip_path = os.path.join(self._root, self.zip_filename)
        self.extract_path = os.path.splitext(self.zip_path)[0]
        self.overwrite = overwrite
        self.logger = logger or getLogger(__name__)

    def download(self):
        """データセットをダウンロードする。

        既にデータセットがダウンロードされている場合、overwriteがFalseの場合はダウンロードをスキップします。
        """
        if self.is_downloaded() and not self.overwrite:
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
        """指定されたURLにGETリクエストを送り、レスポンスを返す。

        :param url: リクエストするURL。
        :return: リクエストの結果得られたレスポンスオブジェクト。
        """
        response = requests.get(url, stream=True)
        response.raise_for_status()  # HTTPエラーを確認
        return response

    def save_response_content(self, response: Response, destination, chunk_size: int = 100 * KIB):
        """レスポンスのコンテンツを指定されたファイルに保存する。

        :param response: 保存するためのレスポンスオブジェクト。
        :param destination: 保存先ファイルのパス。
        :param chunk_size: 保存時のチャンクサイズ（デフォルトは100KiB）。
        """
        if not exists(self.root):
            mkdir(self.root)

        with open(destination, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=chunk_size)):
                if chunk:
                    f.write(chunk)

    def extract(self):
        """ZIPファイルを解凍し、重複したルートフォルダがある場合はまとめる。

        解凍先のディレクトリが既に存在する場合、その内容は保持されます。
        """
        print(f"Unzipping {self.zip_path}...")
        try:
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                # ZIPファイルのトップレベルのフォルダを確認
                top_level_dirs = {os.path.normpath(x).split(os.sep)[0] for x in zip_ref.namelist()}

                if len(top_level_dirs) == 1:
                    # トップレベルに1つのディレクトリだけある場合
                    top_level_dir = next(iter(top_level_dirs))
                    self.extract_path = os.path.join(self._root, top_level_dir)
                    print(f"Extracting into {self.extract_path}...")

                total_files = len(zip_ref.namelist())
                with tqdm(total=total_files, unit='file') as progress:
                    for file in zip_ref.namelist():
                        destination = self._root if len(top_level_dirs) == 1 else self.extract_path
                        zip_ref.extract(file, destination)
                        progress.update(1)
        except BadZipFile:
            print("Error: Bad zip file, extraction failed.")

        print(f"Extracted to {self._root}.")

    @property
    def root(self):
        """ダウンロード先のルートディレクトリを取得する。"""
        return self._root

    def is_downloaded(self):
        """ZIPファイルがダウンロードされているかどうかを確認する。

        :return: ZIPファイルが存在する場合はTrue、それ以外はFalse。
        """
        return os.path.exists(self.zip_path)

    def is_extracted(self):
        """データセットが解凍されているかどうかを確認する。

        :return: 解凍先が存在する場合はTrue、それ以外はFalse。
        """
        return os.path.exists(self.extract_path)

    def __call__(self, on_complete: Callable = None):
        """ダウンロードおよび解凍を実行する。

        :param on_complete: 処理完了後に呼び出す関数（オプション）。
        """
        if self.overwrite or not self.is_downloaded():
            self.download()
        else:
            print("Dataset exists and 'overwrite' is False. No download.")

        if self.overwrite or not self.is_extracted():
            self.extract()
        else:
            print("Dataset exists and 'overwrite' is False. No extract.")

        if on_complete is not None:
            on_complete()


class GoogleDriveDownloader(Downloader):
    """
    Google DriveからファイルをダウンロードするためのDownloaderのサブクラス。

    :param root: ダウンロード先のルートディレクトリ。
    :param file_id: Google DriveのファイルID。
    :param overwrite: 既存のファイルを上書きするかどうか（デフォルトはTrue）。
    :param zip_filename: ZIPファイルの名前（指定がない場合はURLから取得）。
    """

    def __init__(self, root: str, file_id: str, overwrite: bool = True, zip_filename=""):
        """
        GoogleDriveDownloaderのコンストラクタ。

        :param root: ダウンロード先のルートディレクトリ。
        :param file_id: Google DriveのファイルID。
        :param overwrite: 既存のファイルを上書きするかどうか（デフォルトはTrue）。
        :param zip_filename: ZIPファイルの名前（指定がない場合はURLから取得）。
        """
        google_drive_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        self.file_id = file_id
        super().__init__(root, google_drive_url, overwrite, zip_filename)

    def request(self, url: str):
        """Google Driveからのダウンロードのためのリクエストを送信する。

        :param url: リクエストするURL。
        :return: リクエストの結果得られたレスポンスオブジェクト。
        """

        def get_confirm_token(response: Response):
            """ダウンロード確認トークンを取得する。

            :param response: レスポンスオブジェクト。
            :return: トークンが存在する場合はその値、存在しない場合はNone。
            """
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
