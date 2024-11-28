"""Module for caching explanations."""

import glob
import os
from typing import Any, Optional, Union

import torch
from torch import Tensor


class Cache:
    """Abstract class for caching. Methods of this class are static."""

    @staticmethod
    def save(*args, **kwargs) -> None:
        """Save the explanation to the cache."""
        raise NotImplementedError

    @staticmethod
    def load(*args, **kwargs) -> Any:
        """Load the explanation from the cache."""
        raise NotImplementedError

    @staticmethod
    def exists(*args, **kwargs) -> bool:
        """Check if the explanation exists in the cache."""
        raise NotImplementedError


class BatchedCachedExplanations:
    """Utility class for lazy loading and saving batched explanations."""

    def __init__(
        self,
        cache_dir: str,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """Load and save batched explanations.

        Parameters
        ----------
        cache_dir: str
            Directory containing the cached explanations.
        device: Optional[Union[str, torch.device]]
            Device to load the explanations on.

        """
        super().__init__()
        self.cache_dir = cache_dir
        self.device = device

        self.av_filesearch = os.path.join(cache_dir, "*.pt")
        self.files = glob.glob(self.av_filesearch)
        self.batch_size = torch.load(
            self.files[0], map_location=self.device, weights_only=True
        ).shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get the explanation at the specified index.

        Parameters
        ----------
        idx: int
            Index of the explanation.

        Returns
        -------
        torch.Tensor
            The explanation at the specified index.

        """
        assert idx < len(self.files), "Layer index is out of bounds!"
        fl = self.files[idx]
        xpl = torch.load(fl, map_location=self.device, weights_only=True)

        return xpl

    def __len__(self) -> int:
        """Get the number of explanations in the cache.

        Returns
        -------
        int
            Number of explanations in the cache.

        """
        return len(self.files)


class ExplanationsCache(Cache):
    """Class for caching generated explanations at a given path."""

    @staticmethod
    def exists(
        path: str,
        num_id: Optional[Union[str, int]] = None,
    ) -> bool:
        """Check if the explanations exist at the given path.

        Parameters
        ----------
        path: str
            Path to the explanations.
        num_id: Optional[Union[str, int]]
            Number identifier for the explanations.

        Returns
        -------
        bool
            True if the explanations exist, False otherwise.

        """
        av_filesearch = os.path.join(
            path, "*.pt" if num_id is None else f"{num_id}.pt"
        )
        return os.path.exists(path) and len(glob.glob(av_filesearch)) > 0

    @staticmethod
    def save(
        path: str,
        exp_tensors: Tensor,
        num_id: Union[str, int],
    ) -> None:
        """Save the explanations to the given path.

        Parameters
        ----------
        path: str
            Path to save the explanations.
        exp_tensors: Tensor
           Explanations to save.
        num_id: Union[str, int]
            Number identifier for the explanations.

        Returns
        -------
        None

        """
        av_save_fl_path = os.path.join(path, f"{num_id}.pt")
        torch.save(exp_tensors.detach().cpu(), av_save_fl_path)

    @staticmethod
    def load(
        path: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> BatchedCachedExplanations:
        """Load the explanations from the given path.

        Parameters
        ----------
        path: str
            Path to load the explanations.
        device: Optional[Union[str, torch.device]]
            Device to load the explanations on.

        Returns
        -------
        BatchedCachedExplanations
            BatchedCachedExplanations object that can load explanations lazily
            by index.

        """
        if os.path.exists(path):
            xpl_dataset = BatchedCachedExplanations(
                cache_dir=path, device=device
            )
            return xpl_dataset
        else:
            raise RuntimeError(f"Explanations were not found at path {path}")
