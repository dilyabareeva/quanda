"""Module to move a dataset to a device."""

from typing import Any, Sized, Union

import torch


def _move_item_to_device(item: Any, device: Union[str, torch.device]) -> Any:
    """Move every tensor in ``item`` to ``device``, preserving structure.

    Non-tensor leaves (ints, strings, etc.) are returned unchanged.
    Scalar targets coming out of ``TensorDataset`` / custom datasets are
    promoted to tensors on the target device so the returned structure is
    ``.to``-compatible with downstream code.
    """
    if isinstance(item, torch.Tensor):
        return item.to(device)
    if isinstance(item, tuple):
        return tuple(_move_item_to_device(x, device) for x in item)
    if isinstance(item, list):
        return [_move_item_to_device(x, device) for x in item]
    if isinstance(item, dict):
        return {k: _move_item_to_device(v, device) for k, v in item.items()}
    if isinstance(item, (int, float, bool)):
        return torch.tensor(item, device=device)
    return item


class OnDeviceDataset(torch.utils.data.Dataset):
    """Wrapper that moves a dataset's tensors to a target device.

    Handles arbitrary sample structures returned by ``dataset[i]`` —
    single tensor, tuple/list of any length, dict — by recursively moving
    every tensor leaf while leaving non-tensor values untouched.
    """

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
        """Get a sample by index with all its tensors on ``self.device``."""
        return _move_item_to_device(self.dataset[idx], self.device)

    def __len__(self):
        """Get dataset length."""
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        dl = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        return len(dl)
