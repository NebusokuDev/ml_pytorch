import glob
from os import path
from typing import Optional, Callable

from PIL import Image
from torch.utils.data import Dataset

from datasets.downloader import Downloader


class SALICONDataset(Dataset):

    def __init__(self,
                 val_mode: bool = True,
                 image_transform: Optional[Callable] = None,
                 map_transform: Optional[Callable] = None,
                 images_downloader: Optional[Downloader] = None,
                 map_downloader: Optional[Downloader] = None,
                 ):

        self.categories = "val" if val_mode else "train"

        self.image_transform = image_transform
        self.map_transform = map_transform

        self.images_downloader = images_downloader or Downloader("./data/salicon", "", zip_filename="images.zip",
                                                                 overwrite=False)
        self.maps_downloader = map_downloader or Downloader("./data/salicon", "", zip_filename="maps.zip",
                                                            overwrite=False)

        self.images_downloader()
        self.maps_downloader()

        # 画像とマップのペアを取得
        self.image_map_pair_cache = []
        self.cache_image_map_paths()

    def cache_image_map_paths(self):
        for category in self.categories:
            images_dir = self.images_downloader.extract_path
            maps_dir = self.maps_downloader.extract_path

            images_path_list = sorted(glob.glob(path.join(images_dir, category, "*.jpg")))
            maps_path_list = sorted(glob.glob(path.join(maps_dir, category, "*.png")))

            # ペアリング
            for img_path, map_path in zip(images_path_list, maps_path_list):
                if path.basename(img_path) == path.basename(map_path).replace(".png", ".jpg"):
                    self.image_map_pair_cache.append((img_path, map_path))

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
