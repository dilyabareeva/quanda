from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    def __init__(self, *args, **kwargs):
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
        # TODO: should it be a tensor or dataset? For large datasets, storing the whole thing in RAM might be difficult.
        **kwargs,
    ):
        """
        Here include some general steps, incl.:

        1) Universal assertions about the passed arguments, incl. checking that the length of train/test datset and
        explanations match.
        2) Call the _explain method.
        3) Format the output into a unified format for all metrics, possible using some arguments passed in kwargs.

        :param model:
        :param model_id:
        :param cache_dir:
        :param train_dataset:
        :param test_dataset:
        :param explanations:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _explain(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        test_dataset: torch.utils.data.Dataset,
        explanations: torch.utils.data.Dataset,
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
