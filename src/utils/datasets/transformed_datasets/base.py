import random
from typing import Any, Callable, Optional, Sized

import torch
from torch.utils.data.dataset import Dataset


class TransformedDataset(Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        cache_path: str = "./cache",
        cls_idx: Optional[int] = None,
        # If isinstance(subset_idx,int): perturb this class with probability p,
        # if isinstance(subset_idx,List[int]): perturb datapoints with these indices with probability p
        p: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
        sample_fn: Optional[Callable] = None,
        label_fn: Optional[Callable] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.n_classes = n_classes
        self.cls_idx = cls_idx
        self.cache_path = cache_path
        self.p = p
        if sample_fn is not None:
            self.sample_fn = sample_fn
        else:
            self.sample_fn = self._identity
        if label_fn is not None:
            self.label_fn = label_fn
        else:
            self.label_fn = self._identity
        self.seed = seed
        self.rng = random.Random()
        self.rng.seed(self.seed)
        self.samples_to_perturb = []
        for i in range(self.__len__()):
            x, y = dataset[i]
            perturb_sample = (self.cls_idx is None) or (y == self.cls_idx)
            p_condition = (self.rng.random() <= self.p) if self.p < 1.0 else True
            perturb_sample = p_condition and perturb_sample
            if perturb_sample:
                self.samples_to_perturb.append(i)

    def __len__(self) -> int:
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        dl = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        return len(dl)

    def __getitem__(self, index) -> Any:
        x, y = self.dataset[index]
        xx = self.sample_fn(x)
        yy = self.label_fn(y)

        return xx, yy if index in self.samples_to_perturb else x, y

    def _get_original_label(self, index) -> int:
        _, y = self.dataset[index]
        return y

    @staticmethod
    def _identity(x: Any) -> Any:
        return x
