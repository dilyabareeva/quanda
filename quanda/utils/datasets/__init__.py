"""Datasets."""

from quanda.utils.datasets.on_device_dataset import OnDeviceDataset
from quanda.utils.datasets.image_datasets import SingleClassImageDataset

data_wrappers = {
    "OnDeviceDataset": OnDeviceDataset,
    "SingleClassImageDataset": SingleClassImageDataset,
}

__all__ = ["OnDeviceDataset", "SingleClassImageDataset"]
