import torch
from torch.utils.data.dataset import Dataset


class IndexedSubset(Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, indices, return_indices=False):
        self.dataset = dataset
        self.indices = indices
        self.return_indices = return_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        d = self.dataset[self.indices[item]]
        if self.return_indices:
            return d, self.indices[item]
        return d
