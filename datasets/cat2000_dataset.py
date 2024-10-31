from typing import Optional
import os
import requests
import zipfile
import glob
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class Cat2000Dataset(Dataset):
    def __init__(self, categories: Optional[list[str]] = None, transform_module=None, download_path: str = "",
                 download: bool = True):
        if categories is None:
            categories = ["*"]
        self.download_path = download_path
        self.dataset_path = os.path.join(self.download_path, "trainSet", "Stimuli")
        self.categories = categories
        self.transform = transform_module

        if download and not self.is_exist_dataset():
            self.download_dataset()

        # 画像とマップのペアを取得
        self.image_map_pairs = self.get_image_map_paths()

    def is_exist_dataset(self):
        return os.path.exists(self.dataset_path)

    def download_dataset(self):
        """データセットをダウンロードし、必要に応じて解凍"""
        zip_path = os.path.join(self.download_path, "trainSet.zip")
        url = "http://saliency.mit.edu/trainSet.zip"

        if os.path.exists(self.dataset_path):
            print(f"Dataset already exists at {self.dataset_path}, skipping download.")
            return

        print(f"Downloading dataset from {url}...")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # HTTPエラーを確認
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0

            with tqdm(total=total_size, unit='B', unit_scale=True, dynamic_ncols=True) as progress:
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=128):
                        downloaded_size += len(chunk)
                        f.write(chunk)
                        progress.update(len(chunk))

            print("\nDownload completed.")

            # ZIPファイルを解凍
            print(f"Unzipping {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.download_path)
            print(f"Extracted to {self.download_path}.")

        except requests.exceptions.RequestException as e:
            print(f"Error during download: {e}")

        except zipfile.BadZipFile:
            print("Error: Bad zip file.")

    def get_image_map_paths(self):
        result = []
        for category in self.categories:
            # 画像ファイルのパスを取得
            image_paths = glob.glob(os.path.join(self.dataset_path, category, "*.jpg"))
            for image_path in image_paths:
                # ベース名を取得してマップファイルのパスを生成
                base_name = os.path.basename(image_path)
                map_name = base_name.replace(".jpg", "_SaliencyMap.jpg")
                map_path = os.path.join(self.dataset_path, category, "Output", map_name)

                if os.path.exists(map_path):
                    result.append((image_path, map_path))
                else:
                    print(f"Warning: No corresponding map found for {image_path}")
        return result

    def __len__(self):
        return len(self.image_map_pairs)

    def __getitem__(self, idx: int):
        image_path, map_path = self.image_map_pairs[idx]
        image, map_image = self.get_raw_item(image_path, map_path)

        if self.transform:
            image = self.transform(image)
            map_image = self.transform(map_image)

        return image, map_image

    def get_raw_item(self, image_path, map_path):
        image = Image.open(image_path).convert("RGB")
        map_image = Image.open(map_path).convert("RGB")

        return image, map_image

    def __str__(self):
        buffer = []
        for index in range(len(self)):
            image, map_image = self[index]
            buffer.append(f"image: {image.size}, map: {map_image.size}")

        return "\\n".join(buffer)
