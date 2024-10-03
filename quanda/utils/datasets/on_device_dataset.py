from typing import Sized, Union

import torch


class OnDeviceDataset(torch.utils.data.Dataset):
    """Wrapper to move a dataset to a device."""

    def __init__(self, dataset: torch.utils.data.Dataset, device: Union[str, torch.device]):
        """
        Constructor for the OnDeviceDataset class.

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
        data, target = self.dataset[idx]
        return data.to(self.device), torch.tensor(target).to(self.device)

    def __len__(self):
        """
        Not all datasets are sized. If the dataset is not sized,
        we create a DataLoader to get the length.

        Returns
        -------
        int
            Dataset length.
        """
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        dl = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        return len(dl)
