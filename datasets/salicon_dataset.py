import glob
import os
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset

from datasets.downloader import Downloader


class SALICONDataset(Dataset):
    def __init__(self,
                 image_transform: Optional[Callable] = None,
                 map_transform: Optional[Callable] = None,
                 downloader: Optional[Downloader] = None,
                 ):
        self.image_transform = image_transform
        self.map_transform = map_transform
        self.downloader = downloader or Downloader("./data", "")
        self.dataset_path = os.path.join(self.downloader.root, "trainSet", "Stimuli")

        self.downloader(on_complete=self.cache_image_map_paths)

        # 画像とマップのペアを取得
        self.image_map_pair_cache = []

    def cache_image_map_paths(self):
        pass

    def __len__(self):
        return len(self.image_map_pair_cache)

    def __getitem__(self, index: int):
        image_path, map_path = self.image_map_pair_cache[index]
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
