from abc import ABC
from typing import Any, Optional, Union

import torch

from quanda.explainers import aggr_types
from quanda.tasks.aggr_strategies import (
    GlobalAggrStrategy,
    GlobalSelfInfluenceStrategy,
)
from quanda.tasks.base import Task


class GlobalRanking(Task, ABC):
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
        global_method: Union[str, type] = "self-influence",
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
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
        super().__init__(model, train_dataset)
        self.expl_kwargs = expl_kwargs or {}
        self.explainer = (
            None if explainer_cls is None else explainer_cls(model=model, train_dataset=train_dataset, **self.expl_kwargs)
        )

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
                "it should be a subclass of BaseAggregator. When passing a string, it should be one of "
                f"{list(aggr_types.keys() + 'self-influence')}."
            )

    def update(
        self,
        explanations: torch.Tensor,
        return_intermediate: bool = False,
        **kwargs: Any,
    ):
        """
        Used to update the metric with new data.

        Parameters
        ----------
        explanations: torch.Tensor
            The explanations.
        return_intermediate: bool
            Whether to return intermediate results.
        **kwargs: Any
            Additional keyword arguments.
        """
        self.strategy.update(explanations, return_intermediate=return_intermediate, **kwargs)

    def reset(self, *args, **kwargs):
        """
        Reset the metric.

        """

        self.strategy.reset(*args, **kwargs)

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Load the state dictionary.
        """
        self.strategy.load_state_dict(state_dict, *args, **kwargs)

    def state_dict(self, *args, **kwargs):
        """
        Get the state dictionary.
        """
        return self.strategy.state_dict(*args, **kwargs)

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Compute the metric.
        """
        return self.strategy.get_global_rank(*args, **kwargs)
