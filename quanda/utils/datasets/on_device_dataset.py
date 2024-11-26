"""Module to move a dataset to a device."""

from typing import Sized, Union

import torch


class OnDeviceDataset(torch.utils.data.Dataset):
    """Wrapper to move a dataset to a device."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
    ):
        """Construct the OnDeviceDataset class.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to move to the device.
        device : Union[str, torch.device]
            The device to move the dataset to.

        """
        self.dataset = dataset
        self.device = device

    def __getitem__(self, idx):
        """Get a sample by index."""
        data, target = self.dataset[idx]
        return data.to(self.device), torch.tensor(target).to(self.device)

    def __len__(self):
        """Get dataset length."""
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        dl = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        return len(dl)
