from typing import Callable, Literal, Optional

import torch

from src.utils.datasets.transformed_datasets.base import TransformedDataset

ClassToGroupLiterals = Literal["random"]


class SampleTransformationDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        cls_idx: Optional[int] = None,
        p: float = 1.0,
        seed: Optional[int] = None,
        device: str = "cpu",
        sample_fn: Optional[Callable] = None,
    ):

        super().__init__(
            dataset=dataset,
            n_classes=n_classes,
            seed=seed,
            device=device,
            p=p,
            cls_idx=cls_idx,
            sample_fn=sample_fn,
        )
