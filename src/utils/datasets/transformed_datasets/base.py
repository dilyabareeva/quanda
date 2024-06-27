from typing import Callable, List, Optional, Union

import torch
from torch.utils.data.dataset import Dataset


class TransformedDataset(Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        cache_path: str = "./cache",
        subset_idx: Optional[Union[List[int], torch.Tensor]] = None,
        cls_idx: Optional[int] = None,
        # If isinstance(subset_idx,int): perturb this class with probability p,
        # if isinstance(subset_idx,List[int]): perturb datapoints with these indices with probability p
        p: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
        sample_fn: Optional[Union[Callable, str]] = None,
        label_fn: Optional[Union[Callable, str]] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.n_classes = n_classes
        if (cls_idx is not None) and (subset_idx is not None):
            raise ValueError("At least one of cls_idx or subset_idx has to be None")
        if cls_idx is not None:
            subset_idx = []
            for i, (x, y) in enumerate(dataset):
                if y == cls_idx:
                    self.subset_idx.append(i)
            self.subset_idx = torch.tensor(self.subset_idx, device=device)
        else:
            self.subset_idx = subset_idx

        self.p = p
        self.sample_fn = sample_fn if sample_fn is not None else lambda x: x
        self.label_fn = label_fn if label_fn is not None else lambda x: x
        self.seed = seed
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)
        self.samples_to_perturb = []
        for i in range(dataset.__len__()):
            x, y = dataset[i]
            condition1 = self.subset_idx is None
            condition2 = isinstance(self.subset_idx, int) and y == self.subset_idx
            condition3 = isinstance(self.subset_idx, list) and i in self.subset_idx
            condition4 = isinstance(self.subset_idx, torch.Tensor) and i in self.subset_idx
            p_condition = (torch.rand(1, generator=self.generator) <= self.p) if self.p < 1.0 else True
            perturb_sample = condition1 or condition2 or condition3 or condition4
            perturb_sample = p_condition and perturb_sample
            if perturb_sample:
                self.samples_to_perturb.append(i)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        xx = self.sample_fn(x)
        yy = self.label_fn(y)

        return xx, yy if index in self.samples_to_perturb else x, y
