from src.utils.globalization.base import Globalization


class GlobalizationFromExplanations(Globalization):
    def update(self, explanations):
        self.scores += explanations.abs().sum(dim=0)
