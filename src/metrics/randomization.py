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
        cache_dir: str,  # TODO: maybe cache is not the best notation?
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        explanations: torch.utils.data.Dataset,
        explain_fn: Callable,
    ):
        rand_model = RandomizationMetric._randomize_model(model)
        rand_explanations=explain_fn(model=model, **self.kwargs)
        self._evaluate
        raise NotImplementedError

    @abstractmethod
    def _evaluate(
        self,
        model: torch.nn.Module,
        original_explanations: torch.utils.data.Dataset,
        random_explanations: torch.utils.data.Dataset,
    ):
        """
        Used to implement metric-specific logic.
        """

        raise NotImplementedError

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
