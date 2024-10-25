from torch.utils.data import Dataset

from datasets.downloader import Downloader


class SaliconDataset(Dataset):
    def __init__(self, downloader: Downloader):
        self.downloader = downloader

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass