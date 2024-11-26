"""Base class for metrics."""

from abc import ABC, abstractmethod
from typing import Any, Union, List, Optional, Callable

import torch

from quanda.utils.common import get_load_state_dict_func


class Metric(ABC):
    """Base class for metrics."""

    def __init__(
        self,
        model: torch.nn.Module,
        checkpoints: Union[str, List[str]],
        train_dataset: torch.utils.data.Dataset,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
    ):
        """Initialize metric.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        checkpoints : Union[str, List[str]]
            Path to the checkpoint file(s) to be used for the attribution
            computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, by default
            None.

        """
        self.device: Union[str, torch.device]
        self.model: torch.nn.Module = model
        self.checkpoints = (
            checkpoints if isinstance(checkpoints, List) else [checkpoints]
        )
        self.train_dataset: torch.utils.data.Dataset = train_dataset

        # if model has device attribute, use it, otherwise the
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
        """Update the metric with new data.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Any:
        """Compute the metric score.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the metric state."""
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        """Load the metric state.

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
        """Get the metric state.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def load_last_checkpoint(self):
        """Load the model from the checkpoint file.

        Parameters
        ----------
        checkpoint : str
            Path to the checkpoint file.

        """
        self.checkpoints_load_func(self.model, self.checkpoints[-1])
