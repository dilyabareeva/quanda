#!/usr/bin/env python
#  type: ignore

from typing import Callable, List, Optional, Union

import torch
from torch.utils.data.dataset import Dataset


class ToyDataset(Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dataset_id: str,
        class_labels: List[str],
        inverse_transform: Callable = lambda x: x,
        cache_path: str = "./datasets",
        subset_idx=Optional[int, List[int], torch.Tensor[int]],
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
        self.dataset_id = dataset_id
        self.inverse_transform = inverse_transform
        self.class_labels = class_labels
        self.subset_idx = subset_idx
        self.p = p
        self.sample_fn = sample_fn if sample_fn is not None else lambda x: x
        self.label_fn = label_fn if label_fn is not None else lambda x: x
        self.seed = seed
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        xx = self.sample_fn(x)
        yy = self.label_fn(y)

        condition1 = self.subset_idx is None
        condition2 = isinstance(self.subset_idx, int) and y == self.subset_idx
        condition3 = isinstance(self.subset_idx, list) and index in self.subset_idx
        condition4 = isinstance(self.subset_idx, torch.Tensor) and index in self.subset_idx
        p_condition = (torch.rand(1, generator=self.generator) <= self.p) if self.p < 1.0 else True

        perturb_sample = condition1 or condition2 or condition3 or condition4
        perturb_sample = p_condition and perturb_sample

        return xx, yy if perturb_sample else x, y
