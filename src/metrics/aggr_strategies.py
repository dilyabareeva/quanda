import warnings
from functools import lru_cache
from typing import Callable, Optional

import torch

from src.explainers.aggregators import BaseAggregator


class GlobalSelfInfluenceStrategy:

    def __init__(
        self,
        si_fn: Optional[Callable] = None,
    ):
        # TODO: right now si_fn is a Callable that takes no argument and outputs self-influence values
        # TODO: it's not very clear design, can we improve it?

        if si_fn is None:
            raise ValueError(
                "Self-influence function (si_fn) is required for a metric " "with global method 'self-influence'."
            )
        self.si_fn = si_fn

    @lru_cache(maxsize=1)
    def get_global_rank(self):
        return self.si_fn().argsort()

    @staticmethod
    def _si_warning(method_name: str):
        warnings.warn(
            f"{method_name} method is not supported for a metric with global method "
            "'self-influence'. Method call will be ignored. Call 'compute' method to get the final result."
        )

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        self._si_warning("update")

    def reset(self, *args, **kwargs):
        pass

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self._si_warning("load_state_dict")

    def state_dict(self, *args, **kwargs):
        self._si_warning("state_dict")


class GlobalAggrStrategy:
    def __init__(
        self,
        aggr_type: type,
    ):
        self.aggregator = aggr_type()
        self.global_rank: torch.Tensor

        if not isinstance(self.aggregator, BaseAggregator):
            raise ValueError(f"Aggregator type {aggr_type} is not supported.")

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        self.aggregator.update(explanations)

    def get_global_rank(self, *args, **kwargs):
        return self.aggregator.compute().argsort()

    def reset(self, *args, **kwargs):
        self.aggregator.reset()

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.aggregator.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        return self.aggregator.state_dict()
