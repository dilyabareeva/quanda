from abc import ABC

import torch


class ExplanationsAggregator(ABC):
    def __init__(self, training_size: int, *args, **kwargs):
        self.scores = torch.zeros(training_size)

    def update(self, explanations: torch.Tensor):
        raise NotImplementedError

    def get_global_ranking(self) -> torch.Tensor:
        return self.scores.argsort()


class SumAggregator(ExplanationsAggregator):
    def update(self, explanations: torch.Tensor) -> torch.Tensor:
        self.scores += explanations.sum(dim=0)


class AbsSumAggregator(ExplanationsAggregator):
    def update(self, explanations: torch.Tensor) -> torch.Tensor:
        self.scores += explanations.abs().sum(dim=0)
