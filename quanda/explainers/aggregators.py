from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseAggregator(ABC):
    def __init__(self):
        self.scores: Optional[torch.Tensor] = None

    @abstractmethod
    def update(self, explanations: torch.Tensor):
        raise NotImplementedError

    def _validate_explanations(self, explanations: torch.Tensor):
        if self.scores is None:
            self.scores = torch.zeros(explanations.shape[1])

        if explanations.shape[1] != self.scores.shape[0]:
            raise ValueError(f"Explanations shape {explanations.shape} does not match the expected shape {self.scores.shape}")

    def reset(self, *args, **kwargs):
        """
        Used to reset the aggregator state.
        """
        self.scores = None

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Used to load the aggregator state.
        """
        self.scores = state_dict["scores"]

    @property
    def state_dict(self, *args, **kwargs):
        """
        Used to return the metric state.
        """
        return {"scores": self.scores}

    def compute(self) -> torch.Tensor:
        if self.scores is None:
            raise ValueError("No scores to aggregate.")
        return self.scores


class SumAggregator(BaseAggregator):
    def update(self, explanations: torch.Tensor):
        self._validate_explanations(explanations)
        self.scores += explanations.sum(dim=0)


class AbsSumAggregator(BaseAggregator):
    def update(self, explanations: torch.Tensor):
        self._validate_explanations(explanations)
        self.scores += explanations.abs().sum(dim=0)


aggr_types = {
    "sum": SumAggregator,
    "sum_abs": AbsSumAggregator,
}