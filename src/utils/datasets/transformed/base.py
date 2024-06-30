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
        self.device = device

        if sample_fn is not None:
            self.sample_fn = sample_fn
        else:
            self.sample_fn = self._identity
        if label_fn is not None:
            self.label_fn = label_fn
        else:
            self.label_fn = self._identity

        self.seed = seed
        self.rng = random.Random(seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)

        trans_idx = torch.rand(len(self), generator=self.torch_rng) <= self.p
        if self.cls_idx is not None:
            trans_idx *= torch.tensor([self.dataset[s][1] == self.cls_idx for s in range(len(self))], dtype=torch.bool)
        self.transform_indices = torch.where(trans_idx)[0]

    def __len__(self) -> int:
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        dl = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        return len(dl)

    def __getitem__(self, index) -> Any:
        x, y = self.dataset[index]
        xx = self.sample_fn(x)
        yy = self.label_fn(y)

        return (xx, yy) if index in self.transform_indices else (x, y)

    def _get_original_label(self, index) -> int:
        _, y = self.dataset[index]
        return y

    @staticmethod
    def _identity(x: Any) -> Any:
        return x
