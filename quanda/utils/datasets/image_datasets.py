"""Dataset classes for image datasets."""

import torch


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
        return item["image"], int(item["label"])
