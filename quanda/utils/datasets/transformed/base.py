"""Base class for transformed datasets."""

import random
from typing import Any, Callable, List, Optional

import torch
from torch.utils.data.dataset import Dataset


class TransformedDataset(Dataset):
    """Dataset wrapper that applies a transformation to a subset of data."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        dataset_transform: Optional[Callable] = None,
        transform_indices: Optional[List] = None,
        cache_path: str = "./cache",
        cls_idx: Optional[int] = None,
        p: float = 1.0,
        seed: int = 42,
        sample_fn: Optional[Callable] = None,
        label_fn: Optional[Callable] = None,
    ):
        """Construct the the TransformedDataset class.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Base dataset to transform.
        n_classes : int
            Number of classes in the dataset.
        dataset_transform : Optional[Callable], optional
            The default transform of the dataset, defaults to None.
        transform_indices : Optional[List], optional
            Indices to transform, by default None.
        cache_path : str, optional
            Path for cache, defaults to "./cache".
        cls_idx : Optional[int], optional
            Class index to transform instances of, defaults to None.
            If `transform_indices`is given, this parameter is ignored.
        p : float, optional
            The probability of transformation for each instance to transform,
            defaults to 1.0.
        seed : int, optional
            Seed for the random number generator, defaults to 42.
        sample_fn : Optional[Callable], optional
           Transform function to apply to samples, defaults to None.
        label_fn : Optional[Callable], optional
            Transform function to apply to labels, defaults to None.

        """
        # TODO: add validation for cls_idx and transform_indices
        super().__init__()
        self.dataset = dataset
        self.n_classes = n_classes
        self.cls_idx = cls_idx
        self.cache_path = cache_path
        self.p = p

        if dataset_transform is not None:
            self.dataset_transform = dataset_transform
        else:
            self.dataset_transform = self._identity

        if sample_fn is not None:
            self.sample_fn = sample_fn
        else:
            self.sample_fn = self._identity

        if label_fn is not None:
            self.label_fn = label_fn
        else:
            self.label_fn = self._identity

        self.seed = seed
        self.rng = random.Random(seed)
        self.torch_rng = torch.Generator()
        self.torch_rng.manual_seed(seed)

        if transform_indices is None:
            trans_idx = (
                torch.rand(len(self), generator=self.torch_rng) <= self.p
            )
            if self.cls_idx is not None:
                trans_idx *= torch.tensor(
                    [
                        self.dataset[s][1] == self.cls_idx
                        for s in range(len(self))
                    ],
                    dtype=torch.bool,
                )
            self.transform_indices = torch.where(trans_idx)[0].tolist()
        else:
            self.transform_indices = transform_indices

    def __getitem__(self, index) -> Any:
        """Get a sample by index."""
        x, y = self.dataset[index]

        return (
            (self.dataset_transform(self.sample_fn(x)), self.label_fn(y))
            if (index in self.transform_indices)
            else (self.dataset_transform(x), y)
        )

    def _get_original_label(self, index) -> int:
        """Fet original label of a sample by index.

        Parameters
        ----------
        index : int
            Index of the sample.

        Returns
        -------
        int
            Original label of the sample.

        """
        _, y = self.dataset[index]
        return y

    def __len__(self):
        """Get the length of the dataset."""
        if not hasattr(self.dataset, "__len__"):
            raise ValueError(
                "Dataset needs to implement __len__ to use the "
                "TransformedDataset class."
            )
        else:
            return len(self.dataset)

    @staticmethod
    def _identity(x: Any) -> Any:
        """Identity function."""
        return x
