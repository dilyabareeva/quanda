"""Dataset wrapper for label flipping transformation."""

from typing import Callable, Optional

from torch.utils.data import Dataset

from quanda.utils.datasets.transformed.metadata import LabelFlippingMetadata
from .base import TransformedDataset


class LabelFlippingDataset(TransformedDataset):
    """Dataset wrapper that flips labels for a subset of data."""

    metadata_cls: type = LabelFlippingMetadata

    def __init__(
        self,
        dataset: Dataset,
        metadata: LabelFlippingMetadata,
        dataset_transform: Optional[Callable] = None,
    ):
        """Construct the LabelFlippingDataset class.

        Parameters
        ----------
        dataset : Dataset
            Base dataset to transform.
        dataset_transform : Optional[Callable], optional
            Default transform of the dataset, defaults to None.
        metadata : Optional[LabelFlippingMetadata], optional
            Pre-configured metadata instance, defaults to None.

        """
        super().__init__(
            dataset=dataset,
            dataset_transform=dataset_transform,
            metadata=metadata,
        )
        metadata.transform_indices = (
            metadata.transform_indices or metadata.generate_indices(dataset)
        )
        self.transform_indices = metadata.transform_indices
        self.mislabeling_labels = (
            metadata.mislabeling_labels
            or metadata.generate_mislabeling_labels(dataset)
        )
        self.metadata.validate(dataset)

    def __getitem__(self, idx: int):
        """Get a sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        tuple
            Tuple of (sample, label) where label may be flipped if idx is in
            transform_indices.

        """
        sample, label = super().__getitem__(idx)
        if idx in self.transform_indices:
            label = self.mislabeling_labels[idx]
        return sample, label
