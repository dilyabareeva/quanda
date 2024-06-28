import random
from typing import Optional

import torch

from src.utils.datasets.transformed_datasets.base import TransformedDataset


class LabelPoisoningDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        cls_idx: Optional[int] = None,
        p: float = 1.0,  # TODO: decide on default value vis-à-vis subset_idx
        seed: Optional[int] = None,
        device: str = "cpu",
    ):

        super().__init__(dataset=dataset, n_classes=n_classes, seed=seed, device=device, p=p, cls_idx=cls_idx)
        self.poisoned_labels = {}
        for idx in self.samples_to_perturb:
            y = self._get_original_label(idx)
            self.poisoned_labels[idx] = self._poison(y)

    def _poison(self, original_label):
        label_arr = [i for i in range(self.n_classes) if original_label != i]
        label_idx = random.randint(0, len(label_arr))
        return label_arr[label_idx]

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if index in self.samples_to_perturb:
            y = self.poisoned_labels[index]
        return x, y
