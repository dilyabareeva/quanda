from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    def __init__(
        self, model: torch.nn.Module, train_dataset: torch.utils.data.dataset, device: str = "cpu", *args, **kwargs
    ):
        self.device = device
        self.model = model.to(device)
        self.train_dataset = train_dataset

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
    def load_state_dict(self, state_dict: dict, *args, **kwargs):
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
