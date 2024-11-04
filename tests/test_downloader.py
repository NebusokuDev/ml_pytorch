import os
import unittest
import zipfile
from unittest.mock import patch, mock_open, MagicMock

from datasets.downloader import Downloader


class TestDownloader(unittest.TestCase):

    def setUp(self):
        self.url = "https://getsamplefiles.com/download/zip/sample-1.zip"
        self.root = "./data"

    @patch('requests.get')
    def test_download(self, mock_get):
        # モックレスポンスを設定
        mock_response = MagicMock()
        mock_response.iter_content = lambda chunk_size: [b'test_data']
        mock_response.headers = {'content-length': '9'}
        mock_get.return_value = mock_response

        # ダウンロード先ファイル名とダウンロード URL
        downloader = Downloader(url=self.url, root=self.root, overwrite=True)

        # ファイルを書き込む部分もモック化
        with patch('builtins.open', mock_open()) as mocked_file:
            downloader.download()
            # 正しくファイルが書き込まれたかを確認
            mocked_file().write.assert_called_with(b'test_data')

    @patch('os.path.exists', return_value=False)
    @patch('zipfile.ZipFile')
    def test_extract(self, mock_zipfile, mock_exists):
        # 解凍のモック処理を設定
        mock_zip_ref = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip_ref

        # ダウンローダーを初期化して解凍テスト
        downloader = Downloader(url=self.url, root=self.root)
        downloader.extract()

        # ZIP ファイルの `extractall` メソッドが呼ばれたか確認
        mock_zip_ref.extractall.assert_called_once_with(downloader.file_extracted())

    @patch('os.path.exists', return_value=True)
    def test_is_downloaded(self, mock_exists):
        # ダウンローダーの初期化
        downloader = Downloader(url="http://example.com/file.zip", root="./tests.zip")

        # ファイルが存在するかのテスト
        self.assertTrue(downloader.is_downloaded())

    @patch('requests.get', side_effect=Exception('Download error'))
    def test_download_error(self, mock_get):
        # ダウンロードで例外が発生した場合のテスト
        downloader = Downloader(url="http://example.com/file.zip", root="./tests.zip", overwrite=True)

        with patch('builtins.open', mock_open()):
            downloader.download()
            # エラーが発生するためファイルが作成されないことを確認
            self.assertFalse(os.path.exists("./tests.zip"))

    @patch('os.path.exists', return_value=False)
    @patch('zipfile.ZipFile', side_effect=zipfile.BadZipFile)
    def test_extract_bad_zip(self, mock_zipfile, mock_exists):
        # 無効な ZIP ファイルを解凍する際のエラーテスト
        downloader = Downloader(url="http://example.com/file.zip", root="./tests.zip")

        with self.assertLogs(level='ERROR') as log:
            downloader.extract()
            # ログにエラーメッセージが含まれているかを確認
            self.assertIn("無効なZIPファイルです", log.output[0])


if __name__ == '__main__':
    unittest.main()
