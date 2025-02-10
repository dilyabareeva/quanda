"""Dataset wrapper that groups the classes of a dataset into superclasses."""

from typing import Callable, Optional, Literal

import torch

from quanda.utils.datasets.transformed.metadata import LabelGroupingMetadata
from .base import TransformedDataset

ClassToGroupLiterals = Literal["random"]


class LabelGroupingDataset(TransformedDataset):
    """Dataset wrapper that groups the classes into superclasses."""

    metadata_cls: type = LabelGroupingMetadata

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        metadata: LabelGroupingMetadata,
        dataset_transform: Optional[Callable] = None,
    ):
        """Construct the LabelGroupingDataset class.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to group classes.
        dataset_transform : Optional[Callable], optional
            Default transform of the dataset, defaults to None.
        metadata : Optional[LabelGroupingMetadata], optional
            Pre-configured metadata instance, defaults to None.

        """
        super().__init__(
            dataset=dataset,
            dataset_transform=dataset_transform,
            metadata=metadata,
        )

        self.class_to_group = (
            metadata.generate_class_mapping()
            if metadata.class_to_group == "random"
            else metadata.class_to_group
        )
        metadata.transform_indices = (
            metadata.transform_indices or metadata.generate_indices(dataset)
        )
        self.transform_indices = metadata.transform_indices
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
            Tuple of (sample, group_label) where group_label is the superclass
            label for the original class.

        """
        sample, label = super().__getitem__(idx)
        return sample, self.class_to_group[label]

    def get_original_label(self, idx: int):
        """Get the original lable by index.

        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        int
            The superclass label for the original class.

        """
        sample, label = super().__getitem__(idx)
        return label
