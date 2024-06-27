#!/usr/bin/env python
#  type: ignore

from typing import List, Optional, Union

import torch

from src.utils.toy_datasets.base import ToyDataset


class LabelPoisoningDataset(ToyDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        subset_idx: Optional[Union[List[int], torch.Tensor[int]]] = None,
        p: int = 1.0,  # TODO: decide on default value vis-à-vis subset_idx
        seed: int = 42,
        device: str = "cpu",
    ):

        super().__init__(
            dataset=dataset,
            n_classes=n_classes,
            seed=seed,
            device=device,
            p=p,
            subset_idx=subset_idx,
        )
        self.poisoned_labels = {}
        for idx in range(self.perturbed_indices):
            y = self.get_original_label(idx)
            self.poisoned_labels[idx] = self._poison(y)

    def _poison(self, original_label):
        label_arr = [i for i in range(self.n_classes) if original_label != i]
        label_idx = torch.randint(low=0, high=len(label_arr))
        return label_arr[label_idx]

    def _validate_class_to_group(self, class_to_group):
        assert len(class_to_group) == self.n_classes
        assert all([g in self.groups for g in self.class_to_group.values()])

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if index in self.perturbed_indices:
            y = self.poisoned_labels[index]
        return x, y
