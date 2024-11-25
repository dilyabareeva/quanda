from abc import ABC, abstractmethod
from typing import Any, Union, List, Optional, Callable

import torch

from quanda.utils.common import get_load_state_dict_func


class Metric(ABC):
    """
    Base class for metrics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoints: Union[str, List[str]],
        train_dataset: torch.utils.data.Dataset,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
    ):
        """
        Base class for metrics.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        checkpoints : Union[str, List[str]]
            Path to the checkpoint file(s) to be used for the attribution computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, by default None.
        """
        self.device: Union[str, torch.device]
        self.model: torch.nn.Module = model
        self.checkpoints = checkpoints if isinstance(checkpoints, List) else [checkpoints]
        self.train_dataset: torch.utils.data.Dataset = train_dataset

        # if model has device attribute, use it, otherwise use the default device
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

        if checkpoints_load_func is None:
            self.checkpoints_load_func = get_load_state_dict_func(self.device)
        else:
            self.checkpoints_load_func = checkpoints_load_func

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

    def load_last_checkpoint(self):
        """
        Load the model from the checkpoint file.

        Parameters
        ----------
        checkpoint : str
            Path to the checkpoint file.
        """
        self.checkpoints_load_func(self.model, self.checkpoints[-1])
