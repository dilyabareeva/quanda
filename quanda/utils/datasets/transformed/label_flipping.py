"""Dataset wrapper for label flipping transformation."""

from typing import Callable, List, Optional

from torch.utils.data import Dataset

from quanda.utils.datasets.dataset_handlers import (
    HuggingFaceDatasetHandler,
    get_dataset_handler,
)
from quanda.utils.datasets.transformed.base import TransformedDataset
from quanda.utils.datasets.transformed.metadata import LabelFlippingMetadata


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
        self.handler = get_dataset_handler(dataset)
        metadata.transform_indices = (
            metadata.transform_indices or metadata.generate_indices(dataset)
        )
        self.transform_indices = metadata.transform_indices
        self.mislabeling_labels = (
            metadata.mislabeling_labels
            or metadata.generate_mislabeling_labels(dataset)
        )
        self.metadata.validate(dataset)

    def apply_filter(self, filter_indices: List[int]) -> None:
        """Apply a filter and remap mislabeling labels to new positions."""
        old_to_new = {int(old): new for new, old in enumerate(filter_indices)}
        remapped_labels = {
            old_to_new[idx]: label
            for idx, label in self.mislabeling_labels.items()
            if idx in old_to_new
        }
        super().apply_filter(filter_indices)
        self.mislabeling_labels = remapped_labels
        if not isinstance(self.metadata, LabelFlippingMetadata):
            raise TypeError(
                "metadata must be a LabelFlippingMetadata instance."
            )
        self.metadata.mislabeling_labels = remapped_labels

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
        item = self.dataset[idx]
        if idx in self.transform_indices:
            item = self.handler.with_label(item, self.mislabeling_labels[idx])
        if isinstance(self.handler, HuggingFaceDatasetHandler):
            return item
        sample, label = item
        return self.dataset_transform(sample), label
