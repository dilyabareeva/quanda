from abc import ABC, abstractmethod
from typing import Any, Sized, Union

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
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.
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

        Parameters
        ----------
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Used to compute the metric.

        Parameters
        ----------
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The computed metric result dictionary.
        """

        raise NotImplementedError

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any):
        """
        Used to reset the metric.

        Parameters
        ----------
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        None
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

        Returns
        -------
        None
        """

        raise NotImplementedError

    @abstractmethod
    def state_dict(self, *args: Any, **kwargs: Any) -> dict:
        """
        Used to get the metric state.

        Parameters
        ----------
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        dict
            The metric state dictionary.
        """

        raise NotImplementedError

    @property
    def dataset_length(self) -> int:
        """
        Get the length of the dataset.

        Returns
        -------
        int
            The length of the dataset.
        """
        if isinstance(self.train_dataset, Sized):
            return len(self.train_dataset)
        dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=1)
        return len(dl)
