from typing import Callable, Dict, List, Optional

import torch

from quanda.utils.datasets.transformed import TransformedDataset


class LabelFlippingDataset(TransformedDataset):
    """Dataset wrapper that applies poisons a subset of the dataset labels randomly."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        dataset_transform: Optional[Callable] = None,
        transform_indices: Optional[List] = None,
        poisoned_labels: Optional[Dict[int, int]] = None,
        cls_idx: Optional[int] = None,
        p: float = 1.0,  # TODO: decide on default value vis-à-vis subset_idx
        seed: int = 42,
    ):
        """Constructor for the LabelFlippingDataset class.

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
        poisoned_labels : Optional[Dict[int, int]], optional
            Dictionary of indices and poisoned labels. If None, the poisoned labels are generated randomly.
        cls_idx : Optional[int], optional
            Class to poison. If `transform_indices` is given, this parameter is ignored, defaults to None
        p : float, optional
            Probability of transformation for each instance to transform, defaults to 1.0
        """
        super().__init__(
            dataset=dataset,
            n_classes=n_classes,
            dataset_transform=dataset_transform,
            transform_indices=transform_indices,
            seed=seed + 1,
            p=p,
            cls_idx=cls_idx,
        )
        if poisoned_labels is not None:
            self._validate_poisoned_labels(poisoned_labels)
            self.transform_indices = list(poisoned_labels.keys())
            self.poisoned_labels = poisoned_labels
        else:
            self.poisoned_labels = {
                i: self._poison(self.dataset[i][1]) for i in range(len(self)) if i in self.transform_indices
            }

    def _poison(self, original_label):
        """Function that poisons labels"""
        label_arr = [i for i in range(self.n_classes) if original_label != i]
        label_idx = self.rng.randint(0, len(label_arr) - 1)
        return label_arr[label_idx]

    def _validate_poisoned_labels(self, poisoned_labels: Dict[int, int]):
        """Validates the poisoned labels as they are supplied by the user.

        Parameters
        ----------
        poisoned_labels : Dict[int, int]
            Dictionary of indices and poisoned labels.

        Raises
        ------
        ValueError
            If the poisoned_labels are not a dictionary of integer keys and values
        """
        if not isinstance(poisoned_labels, dict):
            raise ValueError(
                f"poisoned_labels should be a dictionary of integer keys and values, received {type(poisoned_labels)}"
            )

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if index in self.transform_indices:
            y = self.poisoned_labels[index]
        return self.dataset_transform(x), y
