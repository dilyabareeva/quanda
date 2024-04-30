from abc import ABC, abstractmethod

import torch


class Metric(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(
        self,
        test_dataset: torch.utils.data.Dataset,
        explanations: torch.utils.data.Dataset,
        **kwargs,
    ):
        """

        1) Universal assertions about the passed arguments, incl. checking that the length of train/test datset and
        explanations match.
        2) Call the _evaluate method.
        3) Format the output into a unified format for all metrics, possible using some arguments passed in kwargs.


        :param test_dataset:
        :param explanations:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate(
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
