from abc import ABC, abstractmethod
from typing import Any, Optional, Sized, Union

import torch

from src.explainers.aggregators import BaseAggregator, aggr_types
from src.explainers.base import BaseExplainer
from src.metrics.aggr_strategies import (
    GlobalAggrStrategy,
    GlobalSelfInfluenceStrategy,
)


class Metric(ABC):
    """
    Base class for metrics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        *args: Any,
        **kwargs: Any,
    ):
        """
        Base class for metrics.

        Parameters
        ----------
        model: torch.nn.Module
            A PyTorch model.
        train_dataset: torch.utils.data.Dataset
            A PyTorch dataset.
        device: str
            Device to use.
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.
        """

        self.model: torch.nn.Module = model.to(device)
        self.train_dataset: torch.utils.data.Dataset = train_dataset
        self.device: str = device

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

    def explain_update(
        self,
        *args,
        **kwargs,
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
        if hasattr(self, "explain_fn"):
            raise NotImplementedError
        raise RuntimeError("Explain function not found in explainer.")

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


class GlobalMetric(Metric, ABC):
    """
    Base class for global metrics.
    """

    strategies = {
        "self-influence": GlobalSelfInfluenceStrategy,
        "aggr": GlobalAggrStrategy,
    }

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        global_method: Union[str, BaseAggregator] = "self-influence",
        explainer: Optional[BaseExplainer] = None,
        expl_kwargs: Optional[dict] = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """
        Base class for global metrics.

        Parameters
        ----------
        model: torch.nn.Module
            A PyTorch model.
        train_dataset: torch.utils.data.Dataset
            A PyTorch dataset.
        global_method: Union[str, BaseAggregator]
            The global method to use. It can be a string "sum", "sum_abs", "self-influence", or a custom aggregator.
        explainer: Optional[BaseExplainer]
            An explainer object.
        expl_kwargs: Optional[dict]
            Keyword arguments for the explainer.
        device: str
            Device to use.
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.
        """
        super().__init__(model, train_dataset, device)

        self.expl_kwargs = expl_kwargs or {}
        self.explainer: Optional[BaseExplainer] = explainer

        if isinstance(global_method, str):

            if global_method == "self-influence":
                self.strategy = self.strategies[global_method](explainer=self.explainer)

            elif global_method in aggr_types:
                aggr_type = aggr_types[global_method]
                self.strategy = self.strategies["aggr"](aggr_type=aggr_type)

            else:
                raise ValueError(f"Global method {global_method} is not supported.")

        elif isinstance(global_method, type):
            self.strategy = self.strategies["aggr"](
                aggr_type=global_method,
            )

        else:
            raise ValueError(
                f"Global method {global_method} is not supported. When passing a custom aggregator, "
                "it should be an instance of BaseAggregator. When passing a string, it should be one of "
                f"{list(aggr_types.keys())}."
            )
