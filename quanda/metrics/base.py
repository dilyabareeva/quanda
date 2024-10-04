from abc import ABC, abstractmethod
from typing import Any, Union

import torch


class Metric(ABC):
    """
    Base class for metrics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
    ):
        """
        Base class for metrics.

        Parameters
        ----------
        model: torch.nn.Module
            A PyTorch model.
        train_dataset: torch.utils.data.Dataset
            A PyTorch dataset.
        """
        self.device: Union[str, torch.device]
        self.model: torch.nn.Module = model
        self.train_dataset: torch.utils.data.Dataset = train_dataset

        # if model has device attribute, use it, otherwise use the default device
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

    @abstractmethod
    def update(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Used to update the metric with new data.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Any:
        """
        Used to compute the metric score.

        Raises
        ------
        NotImplementedError
        """

        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Used to reset the metric state.

        """
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        """
        Used to load the metric state.

        Parameters
        ----------
        state_dict: dict
            The metric state dictionary.

        Raises
        ------
        NotImplementedError
        """

        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        """
        Used to get the metric state.

        Raises
        ------
        NotImplementedError
        """

        raise NotImplementedError
