import glob
import os
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset

from datasets.downloader import Downloader


class Cat2000Dataset(Dataset):
    def __init__(self, categories: Optional[list[str]] = None,
                 image_transform: Optional[Callable] = None,
                 map_transform: Optional[Callable] = None,
                 downloader: Optional[Downloader] = None,
                 ):
        if categories is None:
            categories = ["*"]
        self.categories = categories
        self.image_transform = image_transform
        self.map_transform = map_transform
        self.downloader = downloader or Downloader("./data", "http://saliency.mit.edu/trainSet.zip")
        self.dataset_path = os.path.join(self.downloader.root, "trainSet", "Stimuli")

        self.downloader(on_complete=self.cache_image_map_paths)

        # 画像とマップのペアを取得
        self.image_map_pair_cache = []

    def cache_image_map_paths(self):
        self.image_map_pair_cache = []
        for category in self.categories:
            # 画像ファイルのパスを取得
            image_paths = glob.glob(os.path.join(self.dataset_path, category, "*.jpg"))
            for image_path in image_paths:
                # ベース名を取得してマップファイルのパスを生成
                base_name = os.path.basename(image_path)
                map_name = base_name.replace(".jpg", "_SaliencyMap.jpg")
                map_path = os.path.join(self.dataset_path, category, "Output", map_name)

                if os.path.exists(map_path):
                    self.image_map_pair_cache.append((image_path, map_path))
                else:
                    print(f"Warning: No corresponding map found for {image_path}")

    def __len__(self):
        return len(self.image_map_pair_cache)

    def __getitem__(self, idx: int):
        image_path, map_path = self.image_map_pair_cache[idx]
        image = Image.open(image_path).convert("RGB")
        map_image = Image.open(map_path).convert("RGB")

        if self.image_transform is not None:
            image = self.image_transform(image)

            if self.map_transform is not None:
                map_image = self.map_transform(map_image)
            else:
                map_image = self.image_transform(map_image)

        return image, map_image

    def __str__(self):
        return "\n".join(
            f"image: {Image.open(pair[0]).size}, map: {Image.open(pair[1]).size}" for pair in self.image_map_pair_cache)
