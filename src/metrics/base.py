from abc import ABC, abstractmethod
from typing import Sized

import torch


class Metric(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        self.model: torch.nn.Module = model.to(device)
        self.train_dataset: torch.utils.data.Dataset = train_dataset
        self.device: str = device

    @abstractmethod
    def update(
        self,
        *args,
        **kwargs,
    ):
        """
        Used to update the metric with new data.
        """

        raise NotImplementedError

    @abstractmethod
    def compute(self, *args, **kwargs):
        """
        Used to aggregate current results and return a metric score.
        """

        raise NotImplementedError

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        Used to reset the metric state.
        """

        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        """
        Used to load the metric state.
        """

        raise NotImplementedError

    @abstractmethod
    def state_dict(self, *args, **kwargs):
        """
        Used to return the metric state.
        """

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
