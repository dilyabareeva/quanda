"""Dataset classes for image datasets."""

import torch


_IMAGE_KEYS = ("image", "img", "pixel_values")


class HFtoTV(torch.utils.data.Dataset):
    """Wrapper to make Hugging Face datasets compatible with torchvision."""

    def __init__(self, dataset, transform=None):
        """Construct the HFtoTV dataset."""
        self.dataset = dataset
        self.transform = transform
        sample = dataset[0]
        for key in _IMAGE_KEYS:
            if key in sample:
                self.image_key = key
                break
        else:
            raise ValueError(
                f"Could not find image key in dataset. "
                f"Expected one of {_IMAGE_KEYS}, got {list(sample.keys())}."
            )

    def __len__(self):
        """Get dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get a sample by index."""
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        item = self.dataset[idx]
        img = item[self.image_key]
        if self.transform:
            img = self.transform(img)
        return img, int(item["label"])
