import glob
import os
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset

from datasets.downloader import Downloader


class Cat2000(Dataset):
    def __init__(self, categories: Optional[list[str]] = None,
                 image_transform: Optional = None,
                 map_transform: Optional = None,
                 downloader: Optional[Downloader] = None,
                 ):
        if categories is None:
            categories = ["*"]
        self.categories = categories
        self.image_transform = image_transform
        self.map_transform = map_transform
        self.downloader = downloader or Downloader("./data", "http://saliency.mit.edu/trainSet.zip")
        self.dataset_path = os.path.join(self.downloader.root, "trainSet", "Stimuli")

        # 画像とマップのペアを取得
        self.image_map_pair_cache = []
        self.downloader(on_complete=self.cache_image_map_paths)

    def cache_image_map_paths(self):
        self.image_map_pair_cache = []

        # categoriesにワイルドカードが含まれている場合、全カテゴリディレクトリを展開
        if "*" in self.categories:
            expanded_categories = [d for d in glob.glob(os.path.join(self.dataset_path, "*")) if os.path.isdir(d)]
        else:
            expanded_categories = [os.path.join(self.dataset_path, category) for category in self.categories]

        # 展開したカテゴリディレクトリごとに処理を行う
        for category_path in expanded_categories:
            # 画像ファイルのパスを取得
            image_paths = glob.glob(os.path.join(category_path, "*.jpg"))

            for image_path in image_paths:
                base_name = os.path.basename(image_path)
                map_name = base_name.replace(".jpg", "_SaliencyMap.jpg")
                map_path = os.path.join(category_path, "Output", map_name)

                if os.path.exists(map_path):
                    self.image_map_pair_cache.append((image_path, map_path))
                else:
                    print(f"Warning: No corresponding map found for {map_path}")

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
