from typing import Callable, Literal, Optional

import torch

from quanda.utils.datasets.transformed import TransformedDataset

ClassToGroupLiterals = Literal["random"]


class BackdoorPoisoningDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        poisoned_cls: int,
        backdoor_cls: int,
        sample_fn: Callable,
        dataset_transform: Optional[Callable] = None,
        p: float = 0.3,
        seed: int = 42,
    ):
        super().__init__(
            dataset=dataset,
            n_classes=n_classes,
            dataset_transform=dataset_transform,
            seed=seed,
            p=p,
            cls_idx=poisoned_cls,
            sample_fn=sample_fn,
            label_fn=lambda x: backdoor_cls,
        )
