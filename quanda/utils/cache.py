import glob
import os
from typing import Any, List, Optional, Union

import torch
from captum.attr import LayerActivation  # type: ignore
from torch import Tensor


class Cache:
    """
    Abstract class for caching.
    """

    def __init__(self):
        pass

    @staticmethod
    def save(*args, **kwargs) -> None:
        """ Save the explanation to the cache. """
        raise NotImplementedError

    @staticmethod
    def load(*args, **kwargs) -> Any:
        """ Load the explanation from the cache. """
        raise NotImplementedError

    @staticmethod
    def exists(*args, **kwargs) -> bool:
        """ Check if the explanation exists in the cache. """
        raise NotImplementedError


class BatchedCachedExplanations:
    """
    Utility class for lazy loading and saving batched explanations.
    """
    def __init__(
        self,
        cache_dir: str,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        Utility class for loading and saving batched explanations.

        Parameters
        ----------
        cache_dir: str
            Directory containing the cached explanations
        device: Optional[Union[str, torch.device]]
            Device to load the explanations on
        """
        super().__init__()
        self.cache_dir = cache_dir
        self.device = device

        self.av_filesearch = os.path.join(cache_dir, "*.pt")
        self.files = glob.glob(self.av_filesearch)
        self.batch_size = self[0].shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get the explanation at the specified index.

        Parameters
        ----------
        idx: int
            Index of the explanation

        Returns
        -------
        torch.Tensor
            The explanation at the specified index
        """

        assert idx < len(self.files), "Layer index is out of bounds!"
        fl = self.files[idx]
        xpl = torch.load(fl, map_location=self.device)

        # assert the value's batch size matches the batch size of the class instance
        assert xpl.shape[0] == self.batch_size, ("Batch size of the value does not "
                                                 "match the batch size of the class instance.")

        return xpl

    def __setitem__(self, idx: int, val: torch.Tensor):
        """
        Save the explanation at the specified index.

        Parameters
        ----------
        idx: int
            Index of the explanation
        val: torch.Tensor
            Explanation to save

        Returns
        -------
        torch.Tensor
            The explanation at the specified index
        """

        # assert the value's batch size matches the batch size of the class instance
        assert val.shape[0] == self.batch_size, ("Batch size of the value does not match "
                                                 "the batch size of the class instance.")

        fl = self.files[idx]
        torch.save(val, fl)
        return val

    def __len__(self) -> int:
        """
        Get the number of explanations in the cache.

        Returns
        -------
        int
            Number of explanations in the cache
        """
        return len(self.files)


class ExplanationsCache(Cache):
    """
    Class for caching generated explanations at a given path.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def exists(
        path: str,
        num_id: Optional[Union[str, int]] = None,
    ) -> bool:
        """
        Check if the explanations exist at the given path.

        Parameters
        ----------
        path: str
            Path to the explanations
        num_id: Optional[Union[str, int]]
            Number identifier for the explanations

        Returns
        -------
        bool
            True if the explanations exist, False otherwise
        """
        av_filesearch = os.path.join(path, "*.pt" if num_id is None else f"{num_id}.pt")
        return os.path.exists(path) and len(glob.glob(av_filesearch)) > 0

    @staticmethod
    def save(
        path: str,
        exp_tensors: List[Tensor],
        num_id: Union[str, int],
    ) -> None:
        """
        Save the explanations to the given path.

        Parameters
        ----------
        path: str
            Path to save the explanations
        exp_tensors: List[Tensor]
            List of explanations to save
        num_id: Union[str, int]
            Number identifier for the explanations

        Returns
        -------
        None

        """
        av_save_fl_path = os.path.join(path, f"{num_id}.pt")
        torch.save(exp_tensors, av_save_fl_path)

    @staticmethod
    def load(
        path: str,
        device: Optional[Union[str, torch.device]] = None,
    ) -> BatchedCachedExplanations:
        """
        Load the explanations from the given path.

        Parameters
        ----------
        path: str
            Path to load the explanations
        device: Optional[Union[str, torch.device]]
            Device to load the explanations on

        Returns
        -------
        BatchedCachedExplanations
            BatchedCachedExplanations object that can load explanations lazily by index.
        """
        if os.path.exists(path):
            xpl_dataset = BatchedCachedExplanations(cache_dir=path, device=device)
            return xpl_dataset
        else:
            raise RuntimeError(f"Activation vectors were not found at path {path}")
