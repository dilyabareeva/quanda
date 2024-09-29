import glob

import torch
from PIL import Image  # type: ignore
from torch.utils.data import Dataset


class SingleClassImageDataset(Dataset):
    def __init__(self, root: str, label: int, transform=None, *args, **kwargs):
        self.root = root
        self.label = label
        self.transform = transform

        # find all images in the root directory
        self.filenames = glob.glob(root + "/*.png")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label


class HFtoTV(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        item = self.dataset[idx]
        if self.transform:
            item["image"] = self.transform(item["image"])
        return item["image"], item["label"]
