"""Base class for transformed datasets."""

from abc import ABC
from typing import Any, Callable, List, Optional

import torch
from torch.utils.data.dataset import Dataset

from quanda.utils.common import ds_len
from quanda.utils.datasets.transformed.metadata import DatasetMetadata


class TransformedDataset(Dataset, ABC):
    """Dataset wrapper that applies a transformation to a subset of data."""

    metadata_cls: type = DatasetMetadata

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        metadata: DatasetMetadata,
        dataset_transform: Optional[Callable] = None,
        sample_fn: Optional[Callable] = None,
        label_fn: Optional[Callable] = None,
    ):
        """Construct the TransformedDataset class.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Base dataset to transform.
        metadata : Optional[DatasetMetadata], optional
            Pre-configured metadata instance, defaults to None.
        dataset_transform : Optional[Callable], optional
            The default transform of the dataset, defaults to None.
        sample_fn : Optional[Callable], optional
            The sample transformation function, defaults to None.
        label_fn : Optional[Callable], optional
            The label transformation function, defaults to None.

        """
        # check if dataset has length attribute
        if not hasattr(dataset, "__len__"):
            raise ValueError(
                "Transformed dataset must have a length attribute"
            )

        super().__init__()
        self.dataset = dataset
        self.dataset_transform = dataset_transform or (lambda x: x)
        self.sample_fn = sample_fn or (lambda x: x)
        self.label_fn = label_fn or (lambda x: x)
        self.metadata = metadata
        self.transform_indices = (
            metadata.transform_indices or metadata.generate_indices(dataset)
        )

    def __len__(self) -> int:
        """Get dataset length."""
        return ds_len(self.dataset)

    def __getitem__(self, index) -> Any:
        """Get a sample by index."""
        x, y = self.dataset[index]

        return (
            (self.dataset_transform(self.sample_fn(x)), self.label_fn(y))
            if (index in self.transform_indices)
            else (self.dataset_transform(x), y)
        )

    def apply_filter(self, filter_indices: List[int]) -> None:
        """Apply a filter to the dataset and update the transform indices."""
        # Build a mapping from old index to new index
        old_to_new = {int(old): new for new, old in enumerate(filter_indices)}
        # Remap transform_indices to new positions
        self.transform_indices = [
            old_to_new[idx]
            for idx in self.transform_indices
            if idx in old_to_new
        ]
        # Update metadata
        self.metadata.transform_indices = self.transform_indices
        # Apply the subset
        self.dataset = torch.utils.data.Subset(self.dataset, filter_indices)
