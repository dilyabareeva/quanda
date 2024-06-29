import warnings
from typing import Callable, Optional, Union

import torch

from src.explainers.aggregators import BaseAggregator, aggr_types
from src.metrics.base import Metric
from src.utils.common import class_accuracy, make_func


class DatasetCleaningSI:

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        si_fn: Optional[Callable] = None,
        top_k: int = 50,
        device: str = "cpu",
    ):
        if si_fn is None:
            raise ValueError(
                "Self-influence function (si_fn) is required for DatasetCleaning metric "
                "with global method 'self-influence'."
            )

        self.model = model
        self.train_dataset = train_dataset
        self.top_k = top_k
        self.si_fn = si_fn
        self.device = device

        self.global_rank: torch.Tensor = self.si_fn()

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        warnings.warn(
            "update method is not supported for DatasetCleaning metric with global method "
            "'self-influence'. Method call will be ignored."
        )

    def reset(self, *args, **kwargs):
        pass

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        warnings.warn(
            "load_state_dict method is not supported for DatasetCleaning metric with global "
            "method 'self-influence'.Method call will be ignored."
        )

    def state_dict(self, *args, **kwargs):
        warnings.warn(
            "state_dict method is not supported for DatasetCleaning metric with global method "
            "'self-influence'. Method call will be ignored."
        )


class DatasetCleaningAggr:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        aggr_type: type,
        top_k: int = 50,
        device: str = "cpu",
    ):
        self.aggregator = aggr_type()
        self.global_rank: torch.Tensor

        if not isinstance(self.aggregator, BaseAggregator):
            raise ValueError(f"Aggregator type {aggr_type} is not supported.")

        self.model = model
        self.train_dataset = train_dataset
        self.top_k = top_k
        self.device = device

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):

        self.aggregator.update(explanations)

    def reset(self, *args, **kwargs):
        self.aggregator.reset()

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.aggregator.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        return self.aggregator.state_dict()


class DatasetCleaning(Metric):
    """
    Quote from https://proceedings.mlr.press/v89/khanna19a.html:

    'Our goal in this experiment is to try to identify some such misleading training data points,
    and remove them to see if it improves predictive accuracy. To illustrate the flexibility of
    our approach, we focus only on the digits 4 and 9 in the test data which were misclassified
    by our model, and then select the training data points responsible for those misclassifications.'

    """

    strategies = {
        "self-influence": DatasetCleaningSI,
        "aggr": DatasetCleaningAggr,
    }

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        global_method: Union[str, BaseAggregator] = "self-influence",
        si_fn: Optional[Callable] = None,
        si_fn_kwargs: Optional[dict] = None,
        explain_fn: Optional[Callable] = None,
        explain_fn_kwargs: Optional[dict] = None,
        batch_size: int = 32,
        model_id: str = "0",
        cache_dir: str = "./cache",
        top_k: int = 50,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(model=model, train_dataset=train_dataset, device=device)
        self.top_k = top_k
        self.explain_fn_kwargs = explain_fn_kwargs or {}
        self.si_fn_kwargs = si_fn_kwargs or {}
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.batch_size = batch_size

        self.clean_accuracy: int
        self.original_accuracy: int
        self.explain_fn: Optional[Callable]
        self.si_fn: Optional[Callable]

        if explain_fn is not None:
            self.explain_fn = make_func(
                func=explain_fn,
                model=self.model,
                model_id=self.model_id,
                cache_dir=self.cache_dir,
                train_dataset=self.train_dataset,
                device=self.device,
                batch_size=self.batch_size,
                **self.explain_fn_kwargs,
            )
        else:
            self.explain_fn = None

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
                self.strategy = self.strategies[global_method](
                    model=model, train_dataset=train_dataset, explain_fn=si_fn, top_k=top_k, device=device
                )

            elif global_method in aggr_types:
                aggr_type = aggr_types[global_method]
                self.strategy = self.strategies["aggr"](
                    model=model, train_dataset=train_dataset, aggr_type=aggr_type, top_k=top_k, device=device
                )

            else:
                raise ValueError(f"Global method {global_method} is not supported.")

        elif isinstance(global_method, type):
            self.strategy = self.strategies["aggr"](
                model=model, train_dataset=train_dataset, aggr_type=global_method, top_k=top_k, device=device
            )

        else:
            raise ValueError(
                f"Global method {global_method} is not supported. When passing a custom aggregator, "
                "it should be an instance of BaseAggregator. When passing a string, it should be one of "
                f"{list(aggr_types.keys())}."
            )

    def explain_update(
        self,
        data: torch.Tensor,
        *args,
        **kwargs,
    ):
        if self.explain_fn is not None:
            explanations = self.explain_fn(data, *args, **kwargs)
            self.update(explanations, **kwargs)
        else:
            raise RuntimeError("Explain function not found in explainer.")

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        self.strategy.update(explanations, **kwargs)

    def reset(self, *args, **kwargs):
        self.strategy.reset()

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.strategy.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        return self.strategy.state_dict()

    def compute(self, *args, **kwargs):
        top_k_indices = torch.topk(self.strategy.global_rank, self.top_k).indices
        clean_indices = [i for i in range(self.dataset_length) if i not in top_k_indices]
        clean_subset = torch.utils.data.Subset(self.train_dataset, clean_indices)

        train_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        clean_dl = torch.utils.data.DataLoader(clean_subset, batch_size=32, shuffle=True)

        self.clean_accuracy = class_accuracy(self.model, clean_dl)
        self.original_accuracy = class_accuracy(self.model, train_dl)

        return self.original_accuracy - self.clean_accuracy
