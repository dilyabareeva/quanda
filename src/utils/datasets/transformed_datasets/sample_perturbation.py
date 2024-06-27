from typing import Callable, List, Literal, Optional, Union

import torch

from src.utils.datasets.transformed_datasets.base import TransformedDataset

ClassToGroupLiterals = Literal["random"]


class SamplePerturbationDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        subset_idx: Optional[Union[List[int], torch.Tensor]] = None,
        cls_idx: Optional[int] = None,
        p: float = 1.0,
        seed: int = 42,
        device: str = "cpu",
        sample_fn: Optional[Union[Callable, str]] = None,
    ):

        super().__init__(
            dataset=dataset,
            n_classes=n_classes,
            seed=seed,
            device=device,
            p=p,
            cls_idx=cls_idx,
            subset_idx=subset_idx,
            sample_fn=sample_fn,
            cls_idx=cls_idx,
        )
