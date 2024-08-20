from typing import Callable, List, Literal, Optional

import torch

from quanda.utils.datasets.transformed import TransformedDataset

ClassToGroupLiterals = Literal["random"]


class SampleTransformationDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        dataset_transform: Optional[Callable] = None,
        transform_indices: Optional[List[int]] = None,
        cls_idx: Optional[int] = None,
        p: float = 1.0,
        seed: int = 42,
        sample_fn: Optional[Callable] = None,
    ):
        super().__init__(
            dataset=dataset,
            n_classes=n_classes,
            dataset_transform=dataset_transform,
            transform_indices=transform_indices,
            seed=seed,
            p=p,
            cls_idx=cls_idx,
            sample_fn=sample_fn,
        )
