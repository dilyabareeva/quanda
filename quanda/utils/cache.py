"""Module for caching explanations."""

import glob
import os
from typing import Any, Optional, Union

import torch


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
        device: Optional[str] = None,
    ):
        """Load and save batched explanations.

        Parameters
        ----------
        cache_dir: str
            Directory containing the cached explanations.
        device: Optional[str]
            Device to load the explanations on.

        """
        super().__init__()
        self.cache_dir = cache_dir
        self.device = device

        self.av_filesearch = os.path.join(cache_dir, "*.pt")
        files = glob.glob(self.av_filesearch)

        # Index files by their num_id (filename stem). Numeric stems
        # are stored as ints so int lookups (e.g. batch index) work.
        self._by_id: dict = {}
        for fl in files:
            stem = os.path.splitext(os.path.basename(fl))[0]
            key: Union[int, str] = (
                int(stem) if stem.lstrip("-").isdigit() else stem
            )
            self._by_id[key] = fl

        self.files = [
            self._by_id[k]
            for k in sorted(
                self._by_id.keys(),
                key=lambda x: (isinstance(x, str), x),
            )
        ]
        self.batch_size = torch.load(
            self.files[0], map_location=self.device, weights_only=True
        ).shape[0]

    def keys(self):
        """Return the num_ids available in the cache."""
        return list(self._by_id.keys())

    def __getitem__(self, num_id: Union[int, str]) -> torch.Tensor:
        """Load the explanation tensor saved with the given ``num_id``.

        Parameters
        ----------
        num_id: Union[int, str]
            Identifier the tensor was saved under via
            :meth:`ExplanationsCache.save`.

        Returns
        -------
        torch.Tensor
            The explanation at the specified index.

        """
        if num_id not in self._by_id:
            raise KeyError(
                f"num_id {num_id!r} not found in cache {self.cache_dir}."
            )
        fl = self._by_id[num_id]
        return torch.load(fl, map_location=self.device, weights_only=True)

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
        exp_tensors: torch.Tensor,
        num_id: Union[str, int],
    ) -> None:
        """Save the explanations to the given path.

        Parameters
        ----------
        path: str
            Path to save the explanations.
        exp_tensors: torch.Tensor
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
        device: Optional[str] = None,
    ) -> BatchedCachedExplanations:
        """Load the explanations from the given path.

        Parameters
        ----------
        path: str
            Path to load the explanations.
        device: Optional[str]
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
