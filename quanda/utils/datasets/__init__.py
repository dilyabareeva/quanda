"""Datasets."""

from quanda.utils.datasets.on_device_dataset import OnDeviceDataset

data_wrappers = {
    "OnDeviceDataset": OnDeviceDataset,
}

__all__ = ["OnDeviceDataset"]
