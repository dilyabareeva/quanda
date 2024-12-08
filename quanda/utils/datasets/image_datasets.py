"""Dataset classes for image datasets."""

import glob
import os

import torch
from PIL import Image  # type: ignore
from torch.utils.data import Dataset


class SingleClassImageDataset(Dataset):
    """Dataset class for a single class of images."""

    def __init__(
        self,
        root: str,
        train: bool,
        label: int,
        transform=None,
        *args,
        **kwargs,
    ):
        """Construct the SingleClassImageDataset."""
        self.root = root
        self.label = label
        self.transform = transform
        self.train = train

        # find all images in the root directory
        filenames = []
        for extension in ["*.JPEG", "*.jpeg", "*.jpg", "*.png"]:
            filenames += glob.glob(os.path.join(root, extension))

        self.filenames = filenames

        filenames = sorted(filenames)

        if os.path.exists(os.path.join(root, "train_indices")):
            train_indices = torch.load(os.path.join(root, "train_indices"))
            test_indices = torch.load(os.path.join(root, "test_indices"))
        else:
            randrank = torch.randperm(len(filenames))
            size = int(len(filenames) / 2)
            train_indices = randrank[:size]
            test_indices = randrank[size:]
            torch.save(train_indices, os.path.join(root, "train_indices"))
            torch.save(test_indices, os.path.join(root, "test_indices"))

        if self.train:
            self.filenames = [filenames[i] for i in train_indices]
        else:
            self.filenames = [filenames[i] for i in test_indices]

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
