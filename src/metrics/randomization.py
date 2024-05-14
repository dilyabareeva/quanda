from abc import ABC, abstractmethod
from types import Callable

import torch


class RandomizationMetric(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(
        self,
        model: torch.nn.Module,
        model_id: str,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        explanations: torch.utils.data.Dataset,
        explain_fn: Callable,
        explain_fn_kwargs: dict
    ):
        # Allow for precomputed random explanations?
        rand_model = RandomizationMetric._randomize_model(model)
        return self._evaluate(explanations,explain_fn, explain_fn_kwargs)

    @abstractmethod
    def _evaluate(
        self,
        model: torch.nn.Module,
        explanations: torch.utils.data.Dataset,
    ):
        """
        Used to implement metric-specific logic.
        """

        raise NotImplementedError

    @staticmethod
    def _randomize_model(model):
        return model

    @staticmethod
    @abstractmethod
    def _format(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        explanations: torch.utils.data.Dataset,
    ):
        """
        Format the output of the metric to a predefined format, maybe string?
        """

        raise NotImplementedError
