"""Dataset classes for image datasets."""

import glob

import torch
from PIL import Image  # type: ignore
from torch.utils.data import Dataset


class SingleClassImageDataset(Dataset):
    """Dataset class for a single class of images."""

    def __init__(self, root: str, label: int, transform=None):
        """Construct the SingleClassImageDataset class."""
        self.root = root
        self.label = label
        self.transform = transform

        # find all images in the root directory
        self.filenames = glob.glob(root + "/*.png")

    def __len__(self):
        """Get dataset length."""
        return len(self.filenames)

    def __getitem__(self, idx):
        """Get a sample by index."""
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
