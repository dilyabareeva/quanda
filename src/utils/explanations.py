import glob
import os
from typing import Callable, List, Optional, Tuple, Union

import torch

from utils.cache import TensorCache as TC


class Explanations:
    def __init__(
        self,
        dataset_id: str,
        top_k: int,
        *args,
        **kwargs,
    ):
        self.dataset_id = dataset_id
        self.top_k = top_k

    def __getitem__(self, index: Union[int, slice]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __setitem__(
        self, index: Union[int, slice], val: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class TensorExplanations(Explanations):
    def __init__(
        self,
        dataset_id: str,
        top_k: Optional[int],
        cache_path: str,
        device: str = "cpu",
    ):
        """
        Returns explanations from cache saved as tensors. __getitem__ and __setitem__ methods are used to access the
        explanations on per-sample basis.

        :param dataset_id:
        :param top_k:
        :param cache_dir:
        """
        super().__init__(dataset_id, top_k)
        self.dataset_id = dataset_id
        self.top_k = top_k
        self.cache_path = cache_path
        self.device = device

        # assertions to check if cache_path exists and is a tensor file
        assert os.path.exists(cache_path), f"Cache path {cache_path} does not exist."
        assert os.path.isfile(cache_path), f"Cache path {cache_path} is not a file."
        assert cache_path.endswith(".pt"), f"Cache path {cache_path} is not a tensor file."

        self.xpl = torch.load(cache_path, map_location=self.device)

        # assertions that the explanations length matches top_k, if top_k is provided
        assert top_k is None or top_k == self.xpl.shape[1], "Top_k does not match the number of explanations."

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param idx:
        :return:
        """
        return self.xpl[idx]

    def __setitem__(self, idx: int, val: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param idx:
        :param val:
        :return:
        """
        self.xpl[idx] = val
        return val

    def __len__(self) -> int:
        return self.xpl.shape[0]


class BatchedCachedExplanations(Explanations):
    def __init__(
        self,
        dataset_id: str,
        top_k: Optional[int],
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
        super().__init__(dataset_id, top_k)
        self.dataset_id = dataset_id
        self.top_k = top_k
        self.cache_dir = cache_dir
        self.device = device

        self.av_filesearch = os.path.join(cache_dir, "*.pt")
        self.files = glob.glob(self.av_filesearch)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: add idx type slice

        :param idx:
        :return:
        """

        assert idx < len(self.files), "Layer index is out of bounds!"
        fl = self.files[idx]
        xpl = torch.load(fl, map_location=self.device)
        return xpl

    def __setitem__(self, idx: int, val: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param idx:
        :param val:
        :return:
        """

        fl = self.files[idx]
        torch.save(val, fl)
        return val

    def __len__(self) -> int:
        return len(self.files)


class CachedExplanations:
    def __init__(
        self,
        train_size: int,
        test_size: int,
        cache_batch_size: int = 32,
        cache_dir: str = "./cached_explanations",
    ):
        """
        Returns explanations from cache. Whilethe explanations are saved in cache in batches, __get
        and __setitem__ methods are used to access the explanations on per-sample basis.

        :param train_size:
        :param test_size:
        :param cache_batch_size:
        :param cache_dir:
        """
        self.train_size = train_size
        self.test_size = test_size
        self.cache_batch_size = cache_batch_size
        self.cache_dir = cache_dir
        self.cache_file_count = 0
        self.explanation_targets = torch.empty(0)
        self.index_count = 0
        self.explanations = torch.tensor(shape=(0, train_size))

    def add(self, explanation_targets: torch.Tensor, explanations: torch.Tensor):
        assert (
            len(explanations.shape) == 3
        ), f"Explanations object has {len(explanations.shape)} dimensions, should be 2 (test_datapoints x training_datapoints)"
        assert explanations.shape[-1] == len(
            self.train_dataset
        ), f"Given explanations are {explanations.shape[-1]} dimensional. This should be the number of training datapoints {len(self.train_dataset)} "
        assert (
            not self.index_count == self.test_size
        ), f"Whole {self.test_size} datapoint explanations are already added. Increase test_size to add new explanaitons."
        explanation_count = explanations.shape[0]
        self.explanations = torch.cat((self.explanations, explanations), dim=0)
        self.explanation_targets = torch.cat((self.explanation_targets, explanation_targets), dim=0)
        self.index_count += explanation_count

        # We need to save the final tensor if we saw as many explanations as the test dataset
        # last_save is a boolean that will tell the save_temp_explanations to save file
        # even if the batch size is not reached, for the last batch (it is false for other batches)
        last_save = self.index_count == self.test_size
        if self.cache:
            self.save_temp_explanations(save_always=last_save)

    def save_temp_explanations(self, save_always: bool = False):
        if save_always or self.explanations.shape[0] > self.cache_batch_size - 1:
            save_tensor = self.explanations[: self.cache_batch_size]
            self.cache_file_count += 1
            TC.save(self.cache_dir, f"explanations_{self.cache_file_count}", save_tensor)
            TC.save(self.cache_dir, f"targets_{self.cache_file_count}")
            self.explanations = self.explanations[self.cache_batch_size :]

    def load_all_explanations(self):
        pass

    def __getitem__(self, index: Union[int, slice]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Returns (explanation, explanation_target)
        if self.cache:
            if type(index) is int:
                return self._getitem_single(index)
            else:
                return self._getitem_slice(index)
        else:
            return self.explanations[index], self.explanation_targets[index]

    def _getitem_single(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id = int(index / self.cache_batch_size)
        leftover_indices = index - file_id * self.cache_batch_size
        explanations = TC.load(self.cache_dir, f"explanations_{file_id}")
        targets = TC.load(self.cache_dir, f"targets_{file_id}")
        return explanations[leftover_indices], targets[leftover_indices]

    def _getitem_slice(self, index: slice) -> Tuple[torch.Tensor, torch.Tensor]:
        ret_exp = torch.empty((0, self.train_size))
        ret_target = torch.empty((0,))
        indices_to_get = Explanations.compute_indices_from_slice(index, self.cache_batch_size)
        for file_id, line_ids in indices_to_get:
            explanations = TC.load(self.cache_dir, f"explanations_{file_id}")
            targets = TC.load(self.cache_dir, f"targets_{file_id}")
            ret_exp = torch.cat((ret_exp, explanations[line_ids]), dim=0)
            ret_target = torch.cat((ret_target, targets[line_ids]), dim=0)
        return ret_exp, ret_target

    def __setitem__(
        self, index: Union[int, slice], val: Tuple[torch.Tensor, Union[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache:
            if type(index) is int:
                self._setitem_single(self, index, val)
            else:
                self._setitem_slice(self, index, val)
        else:
            explanation, target = val
            self.explanations[index] = explanation
            self.explanation_targets[index] = target

    def _setitem_single(
        self, index: int, val: Tuple[torch.Tensor, Union[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        explanation, target = val
        file_id = int(index / self.cache_batch_size)
        leftover_indices = index - file_id * self.cache_batch_size
        explanations = TC.load(self.cache_dir, f"explanations_{file_id}")
        targets = TC.load(self.cache_dir, f"targets_{file_id}")
        explanations[leftover_indices] = explanation
        targets[leftover_indices] = target
        TC.save(self.cache_dir, f"explanations_{file_id}", explanations)
        TC.save(self.cache_dir, f"targets_{file_id}", targets)

    def _setitem_slice(
        self, index: int, val: Tuple[torch.Tensor, Union[torch.Tensor, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        explanation, target = val
        indices_to_get = Explanations.compute_indices_from_slice(index, self.cache_batch_size)
        for file_id, line_ids in indices_to_get:
            explanations = TC.load(self.cache_dir, f"explanations_{file_id}")
            targets = TC.load(self.cache_dir, f"targets_{file_id}")
            explanations[line_ids] = explanation
            targets[line_ids] = target
            TC.save(self.cache_dir, f"explanations_{file_id}", explanations)
            TC.save(self.cache_dir, f"targets_{file_id}", targets)

    @staticmethod
    def compute_indices_from_slice(indices, cache_batch_size):
        id_dict = {id: [] for id in range(indices)}
        pass
