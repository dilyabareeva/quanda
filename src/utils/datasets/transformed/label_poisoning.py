from typing import Optional

import torch

from src.utils.datasets.transformed.base import TransformedDataset


class LabelPoisoningDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        cls_idx: Optional[int] = None,
        p: float = 1.0,  # TODO: decide on default value vis-à-vis subset_idx
        seed: int = 42,
        device: str = "cpu",
    ):

        super().__init__(dataset=dataset, n_classes=n_classes, seed=seed, device=device, p=p, cls_idx=cls_idx)
        self.poisoned_labels = {i: self._poison(self.dataset[i][1]) for i in range(len(self)) if i in self.transform_indices}

    def _poison(self, original_label):
        label_arr = [i for i in range(self.n_classes) if original_label != i]
        label_idx = self.rng.randint(0, len(label_arr))
        return label_arr[label_idx]

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if index in self.transform_indices:
            y = self.poisoned_labels[index]
        return x, y
