"""Dataset wrapper that groups the classes of a dataset into superclasses."""

from typing import Callable, Dict, Optional

import torch

from quanda.utils.datasets.transformed.base import TransformedDataset
from quanda.utils.datasets.transformed.metadata import LabelGroupingMetadata


class LabelGroupingDataset(TransformedDataset):
    """Dataset wrapper that groups the classes into superclasses."""

    metadata_cls: type = LabelGroupingMetadata

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        metadata: LabelGroupingMetadata,
        class_to_group: Dict[int, int],
        n_classes: Optional[int] = None,
        n_groups: Optional[int] = None,
        dataset_transform: Optional[Callable] = None,
    ):
        """Construct the LabelGroupingDataset class.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to group classes.
        metadata : LabelGroupingMetadata
            Per-dataset metadata (transform indices, p, seed, cls_idx).
        class_to_group : Dict[int, int]
            Resolved mapping from original class to group. Shared across
            train/val/eval datasets.
        n_classes : Optional[int]
            Number of original classes. Defaults to ``len(class_to_group)``.
        n_groups : Optional[int]
            Number of target groups. Defaults to the count of distinct
            values in ``class_to_group``.
        dataset_transform : Optional[Callable]
            Default transform of the dataset.

        """
        super().__init__(
            dataset=dataset,
            dataset_transform=dataset_transform,
            metadata=metadata,
        )

        self.class_to_group = dict(class_to_group)
        self.n_classes = (
            n_classes if n_classes is not None else len(self.class_to_group)
        )
        self.n_groups = (
            n_groups
            if n_groups is not None
            else len(set(self.class_to_group.values()))
        )
        metadata.transform_indices = (
            metadata.transform_indices or metadata.generate_indices(dataset)
        )
        self.transform_indices = metadata.transform_indices
        self.metadata.validate(dataset)
        if len(self.class_to_group) != self.n_classes:
            raise ValueError(
                f"Length of class_to_group ({len(self.class_to_group)}) "
                f"does not match n_classes ({self.n_classes})"
            )

    def __getitem__(self, idx: int):
        """Get a sample by index.

        Returns
        -------
        tuple
            ``(sample, group_label)`` where ``group_label`` is the
            superclass label for the original class.

        """
        sample, label = super().__getitem__(idx)
        return sample, self.class_to_group[label]

    def get_original_label(self, idx: int):
        """Get the original label by index."""
        sample, label = super().__getitem__(idx)
        return label
