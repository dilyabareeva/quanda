import torch

from src.utils.globalization.base import Globalization


class GlobalizationFromExplanations(Globalization):
    def update(self, explanations: torch.Tensor):
        self.scores += explanations.abs().sum(dim=0)
