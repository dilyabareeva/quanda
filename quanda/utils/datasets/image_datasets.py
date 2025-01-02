"""Dataset classes for image datasets."""

import glob
import os
from typing import Optional, List

import torch
from PIL import Image  # type: ignore
from torch.utils.data import Dataset


class SingleClassImageDataset(Dataset):
    """Dataset class for a single class of images."""

    def __init__(
        self,
        root: str,
        label: int,
        indices: Optional[List[int]] = None,
        transform=None,
    ):
        """Construct the SingleClassImageDataset."""
        self.root = root
        self.label = label
        self.transform = transform
        self.indices = indices

        # find all images in the root directory
        filenames = []
        for extension in ["*.JPEG", "*.jpeg", "*.jpg", "*.png"]:
            filenames += glob.glob(os.path.join(root, extension))

        self.filenames = sorted(filenames)

    def __len__(self):
        """Get dataset length."""
        if self.indices is None:
            return len(self.filenames)
        return len(self.indices)

    def __getitem__(self, idx):
        """Get a sample by index."""
        if self.indices is not None:
            idx = self.indices[idx]
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label


class HFtoTV(torch.utils.data.Dataset):
    """Wrapper to make Hugging Face datasets compatible with torchvision."""

    def __init__(self, dataset, transform=None):
        """Construct the HFtoTV dataset."""
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """Get dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a sample by index."""
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        item = self.dataset[idx]
        if self.transform:
            item["image"] = self.transform(item["image"])
        return item["image"], item["label"]
