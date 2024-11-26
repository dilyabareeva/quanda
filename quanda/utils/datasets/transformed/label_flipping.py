"""Label flipping dataset wrapper."""

from typing import Callable, Dict, List, Optional

import torch

from quanda.utils.datasets.transformed import TransformedDataset


class LabelFlippingDataset(TransformedDataset):
    """Dataset wrapper that poisons a subset of the dataset labels randomly."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        dataset_transform: Optional[Callable] = None,
        transform_indices: Optional[List] = None,
        mislabeling_labels: Optional[Dict[int, int]] = None,
        cls_idx: Optional[int] = None,
        p: float = 1.0,  # TODO: decide on default value vis-à-vis subset_idx
        seed: int = 42,
    ):
        """Construct the the LabelFlippingDataset class.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset to poison.
        n_classes : int
            Number of classes in the dataset.
        dataset_transform : Optional[Callable], optional
            Default transform of the dataset, defaults to None
        transform_indices : Optional[List], optional
            Indices to transform, defaults to None.
        mislabeling_labels : Optional[Dict[int, int]], optional
            Dictionary of indices and poisoned labels. If None, the poisoned
            labels are generated randomly.
        cls_idx : Optional[int], optional
            Class to poison. If `transform_indices` is given, this parameter is
            ignored, defaults to None.
        p : float, optional
            Probability of transformation for each instance to transform,
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
        )
        if mislabeling_labels is not None:
            self._validate_mislabeling_labels(mislabeling_labels)
            self.transform_indices = list(mislabeling_labels.keys())
            self.mislabeling_labels = mislabeling_labels
        else:
            self.mislabeling_labels = {
                i: self._poison(self.dataset[i][1])
                for i in range(len(self))
                if i in self.transform_indices
            }

    def _poison(self, original_label):
        """Poisons labels."""
        label_arr = [i for i in range(self.n_classes) if original_label != i]
        label_idx = self.rng.randint(0, len(label_arr) - 1)
        return label_arr[label_idx]

    def _validate_mislabeling_labels(self, mislabeling_labels: Dict[int, int]):
        """Validate the poisoned labels as they are supplied by the user.

        Parameters
        ----------
        mislabeling_labels : Dict[int, int]
            Dictionary of indices and poisoned labels.

        Raises
        ------
        ValueError
            If the mislabeling_labels are not a dictionary of integer keys and
            values.

        """
        if not isinstance(mislabeling_labels, dict):
            raise ValueError(
                f"mislabeling_labels should be a dictionary of integer keys "
                f"and values, received {type(mislabeling_labels)}"
            )

    def __getitem__(self, index):
        """Get the item at the specified index."""
        x, y = self.dataset[index]
        if index in self.transform_indices:
            y = self.mislabeling_labels[index]
        return self.dataset_transform(x), y
