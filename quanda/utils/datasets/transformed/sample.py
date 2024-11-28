"""Module for the SampleTransformationDataset class."""

from typing import Callable, List, Optional

import torch

from quanda.utils.datasets.transformed import TransformedDataset


class SampleTransformationDataset(TransformedDataset):
    """Dataset wrapper for sample-specific transformation function."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        sample_fn: Callable,
        dataset_transform: Optional[Callable] = None,
        transform_indices: Optional[List[int]] = None,
        cls_idx: Optional[int] = None,
        p: float = 1.0,
        seed: int = 42,
    ):
        """Construct the SampleTransformationDataset class.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The base dataset to transform.
        n_classes : int
            Number of classes in the dataset.
        sample_fn : Callable
            Transform function to apply to samples.
        dataset_transform : Optional[Callable], optional
            The default transform of the dataset, defaults to None.
        transform_indices : Optional[List[int]], optional
            Indices to transform, by default None.
        cls_idx : Optional[int], optional
            Class index to transform instances of, defaults to None.
            If `transform_indices`is given, this parameter is ignored.
        p : float, optional
            The probability of transformation for each instance to transform,
            defaults to 1.0.
        seed : int, optional
            Seed for the random number generator, defaults to 42.

        """
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
