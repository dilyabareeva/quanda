from abc import ABC, abstractmethod
from typing import List, Optional, Sized, Union

import torch

from quanda.utils.datasets import OnDeviceDataset
from quanda.utils.common import cache_result
from quanda.utils.validation import validate_1d_tensor_or_int_list


class BaseExplainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        cache_dir: Optional[str],
        train_dataset: torch.utils.data.Dataset,
        model_id: Optional[str] = None,
        **kwargs,
    ):
        self.device: Union[str, torch.device]
        self.model = model

        # if model has device attribute, use it, otherwise use the default device
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

        self.model_id = model_id
        self.cache_dir = cache_dir

        # if dataset return samples not on device, move them to device
        if train_dataset[0][0].device != self.device:
            train_dataset = OnDeviceDataset(train_dataset, self.device)

        self.train_dataset = train_dataset

    @abstractmethod
    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        raise NotImplementedError

    @property
    def dataset_length(self) -> int:
        """
        By default, the Dataset class does not always have a __len__ method.
        :return:
        """
        if isinstance(self.train_dataset, Sized):
            return len(self.train_dataset)
        dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=1)
        return len(dl)

    def _process_targets(self, targets: Optional[Union[List[int], torch.Tensor]]):
        if targets is not None:
            # TODO: move validation logic outside at a later point
            validate_1d_tensor_or_int_list(targets)
            if isinstance(targets, list):
                targets = torch.tensor(targets)
            targets = targets.to(self.device)
        return targets

    @cache_result
    def self_influence(self, batch_size: int = 32) -> torch.Tensor:
        """
        Base class implements computing self influences by explaining the train dataset one by one

        :param batch_size:
        :param kwargs:
        :return:
        """

        # Pre-allcate memory for influences, because torch.cat is slow
        influences = torch.empty((self.dataset_length,), device=self.device)
        ldr = torch.utils.data.DataLoader(self.train_dataset, shuffle=False, batch_size=batch_size)

        for i, (x, y) in zip(range(0, self.dataset_length, batch_size), ldr):
            explanations = self.explain(test=x.to(self.device), targets=y.to(self.device))
            influences[i : i + batch_size] = explanations.diag(diagonal=i)

        return influences
