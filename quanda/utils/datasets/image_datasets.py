"""Dataset classes for image datasets."""

from typing import Optional

import torch

_IMAGE_KEYS = ("image", "img", "pixel_values")


class HFtoTV(torch.utils.data.Dataset):
    """Wrapper to make Hugging Face datasets compatible with torchvision."""

    def __init__(
        self, dataset, transform=None, label_override: Optional[int] = None
    ):
        """Construct the HFtoTV dataset."""
        self.dataset = dataset
        self.transform = transform
        self.label_override = label_override
        self._labels_cache: Optional[list] = None
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
        label = (
            self.label_override
            if self.label_override is not None
            else int(item["label"])
        )
        return img, label

    def get_label(self, idx):
        """Return the label at ``idx`` without decoding the image."""
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        if self.label_override is not None:
            return self.label_override
        # Column-only access bypasses HF's lazy Image decoder.
        if self._labels_cache is None:
            self._labels_cache = self.dataset["label"]
        return int(self._labels_cache[idx])
