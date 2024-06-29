from abc import ABC, abstractmethod
from typing import Callable, Optional, Sized, Union

import torch

from src.explainers.aggregators import BaseAggregator, aggr_types
from src.metrics.aggr_strategies import GlobalAggrStrategy, GlobalSIStrategy
from src.utils.common import make_func


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

    def explain_update(
        self,
        *args,
        **kwargs,
    ):
        """
        Used to update the metric with new data.
        """
        if hasattr(self, "explain_fn"):
            raise NotImplementedError
        raise RuntimeError("Explain function not found in explainer.")

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


class GlobalMetric(Metric, ABC):

    strategies = {
        "self-influence": GlobalSIStrategy,
        "aggr": GlobalAggrStrategy,
    }

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        global_method: Union[str, BaseAggregator] = "self-influence",
        si_fn: Optional[Callable] = None,
        si_fn_kwargs: Optional[dict] = None,
        batch_size: int = 32,
        model_id: str = "0",
        cache_dir: str = "./cache",
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(model, train_dataset, device, *args, **kwargs)

        self.si_fn_kwargs = si_fn_kwargs or {}
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.batch_size = batch_size

        self.si_fn: Optional[Callable]

        if si_fn is not None:
            self.si_fn = make_func(
                func=si_fn,
                model=self.model,
                model_id=self.model_id,
                cache_dir=self.cache_dir,
                train_dataset=self.train_dataset,
                device=self.device,
                batch_size=self.batch_size,
                **self.si_fn_kwargs,
            )
        else:
            self.si_fn = None

        if isinstance(global_method, str):

            if global_method == "self-influence":
                self.strategy = self.strategies[global_method](si_fn=self.si_fn)

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
