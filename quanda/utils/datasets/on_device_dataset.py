import torch
from typing import Union


class OnDeviceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, device: Union[str, torch.device]):
        self.dataset = dataset
        self.device = device

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data.to(self.device), torch.tensor(target).to(self.device)

    def __len__(self):
        return len(self.dataset)