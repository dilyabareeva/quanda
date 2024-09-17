from typing import Callable, List, Literal, Optional

import torch

from quanda.utils.datasets.transformed import TransformedDataset

ClassToGroupLiterals = Literal["random"]


# THIS DATASET IS NOT YET USED
# JUST KEEPING HERE INSTEAD OF DELETING ALREADY WRITTEN CODE
class BackdoorPoisoningDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        sample_fn: Callable,
        backdoor_cls: int,
        poisoned_cls: Optional[int] = None,
        transform_indices: Optional[List] = None,
        dataset_transform: Optional[Callable] = None,
        p: float = 0.3,
        seed: int = 42,
    ):
        # TODO: add validation for poisoned_cls and transform_indices
        super().__init__(
            dataset=dataset,
            n_classes=n_classes,
            dataset_transform=dataset_transform,
            seed=seed,
            p=p,
            transform_indices=transform_indices,
            cls_idx=poisoned_cls,
            sample_fn=sample_fn,
            label_fn=lambda x: backdoor_cls,
        )
