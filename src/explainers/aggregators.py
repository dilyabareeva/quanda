from abc import ABC, abstractmethod

import torch


class ExplanationsAggregator(ABC):
    def __init__(self, training_size: int, *args, **kwargs):
        self.scores = torch.zeros(training_size)

    @abstractmethod
    def update(self, explanations: torch.Tensor):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        """
        Used to reset the aggregator state.
        """
        self.scores = torch.zeros_like(self.scores)

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Used to load the aggregator state.
        """
        self.scores = state_dict["scores"]

    def state_dict(self, *args, **kwargs):
        """
        Used to return the metric state.
        """
        return {"scores": self.scores}

    def compute(self) -> torch.Tensor:
        return self.scores.argsort()


class SumAggregator(ExplanationsAggregator):
    def update(self, explanations: torch.Tensor) -> torch.Tensor:
        self.scores += explanations.sum(dim=0)


class AbsSumAggregator(ExplanationsAggregator):
    def update(self, explanations: torch.Tensor) -> torch.Tensor:
        self.scores += explanations.abs().sum(dim=0)
