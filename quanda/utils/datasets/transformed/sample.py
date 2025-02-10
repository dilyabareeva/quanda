"""Module for the SampleTransformationDataset class."""

from typing import Callable, Optional

import torch

from quanda.utils.datasets.transformed.metadata import (
    SampleTransformationMetadata,
)
from .base import TransformedDataset


class SampleTransformationDataset(TransformedDataset):
    """Dataset wrapper for sample-specific transformation function."""

    metadata_cls: type = SampleTransformationMetadata

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        sample_fn: Callable,
        metadata: SampleTransformationMetadata,
        dataset_transform: Optional[Callable] = None,
    ):
        """Construct the SampleTransformationDataset class.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The base dataset to transform.
        sample_fn : Callable
            Transform function to apply to samples.
        dataset_transform : Optional[Callable], optional
            The default transform of the dataset, defaults to None.
        metadata : Optional[SampleTransformationMetadata], optional
            Pre-configured metadata instance, defaults to None.

        """
        super().__init__(
            dataset=dataset,
            dataset_transform=dataset_transform,
            metadata=metadata,
        )

        self.metadata = metadata
        self.transform_indices = self.metadata.generate_indices(dataset)
        self.sample_fn = sample_fn
        self.metadata.validate(dataset)
