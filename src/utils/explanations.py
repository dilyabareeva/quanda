import glob
import os
from typing import Optional, Tuple, Union

import torch


class Explanations:
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """

        Exaplanations interface class. Used to define the interface for the Explanations classes.
        Each explanation class implements __getitem__, __setitem__, and __len__ methods, whereby an "item" is a
        explanation tensor batch.
        :param args:
        :param kwargs:
        """
        pass

    def __getitem__(self, index: Union[int, slice]) -> torch.Tensor:
        raise NotImplementedError

    def __setitem__(self, index: Union[int, slice], val: Tuple[torch.Tensor, torch.Tensor]):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class TensorExplanations(Explanations):
    def __init__(
        self,
        tensor: torch.Tensor,
        batch_size: Optional[int] = 8,
        device: str = "cpu",
    ):
        """
        Returns explanations from cache saved as tensors. __getitem__ and __setitem__ methods are used to access the
        explanations on a batch basis.

        :param dataset_id:
        :param top_k:
        :param cache_dir:
        """
        super().__init__()
        self.device = device
        self.xpl = tensor.to(self.device)
        self.batch_size = batch_size

        # assert the number of explanation dimensions is 2 and insert extra dimension to emulate batching
        assert len(self.xpl.shape) == 2, "Explanations object has more than 2 dimensions."

    def __getitem__(self, idx: Union[int, slice]) -> torch.Tensor:
        """

        :param idx:
        :return:
        """
        return self.xpl[idx * self.batch_size : min((idx + 1) * self.batch_size, self.xpl.shape[0])]

    def __setitem__(self, idx: Union[int, slice], val: Tuple[torch.Tensor, torch.Tensor]):
        """

        :param idx:
        :param val:
        :return:
        """

        self.xpl[idx * self.batch_size : (idx + 1) * self.batch_size] = val
        return val

    def __len__(self) -> int:
        return int(self.xpl.shape[0] // self.batch_size) + 1


class BatchedCachedExplanations(Explanations):
    def __init__(
        self,
        cache_dir: str = "./batch_wise_cached_explanations",
        device: str = "cpu",
    ):
        """
        Returns batched explanations from cache. __getitem__ and __setitem__ methods are used to access the explanations
        on per-batch basis.

        :param dataset_id:
        :param top_k:
        :param cache_dir:
        """
        super().__init__()
        self.cache_dir = cache_dir
        self.device = device

        self.av_filesearch = os.path.join(cache_dir, "*.pt")
        self.files = glob.glob(self.av_filesearch)
        self.batch_size = self[0].shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        TODO: add idx type slice

        :param idx:
        :return:
        """

        assert idx < len(self.files), "Layer index is out of bounds!"
        fl = self.files[idx]
        xpl = torch.load(fl, map_location=self.device)

        # assert the value's batch size matches the batch size of the class instance
        assert (
            xpl.shape[0] == self.batch_size
        ), "Batch size of the value does not match the batch size of the class instance."

        return xpl

    def __setitem__(self, idx: int, val: Tuple[torch.Tensor, torch.Tensor]):
        """

        :param idx:
        :param val:
        :return:
        """

        # assert the value's batch size matches the batch size of the class instance
        assert (
            val.shape[0] == self.batch_size
        ), "Batch size of the value does not match the batch size of the class instance."

        fl = self.files[idx]
        torch.save(val, fl)
        return val

    def __len__(self) -> int:
        return len(self.files)
