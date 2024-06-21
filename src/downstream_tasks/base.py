from abc import ABC, abstractmethod

import torch


class DownstreamTaskEval(ABC):
    def __init__(self, device: str = "cpu", *args, **kwargs):
        """
        I think here it would be nice to pass a general receipt for the downstream task construction.
        For example, we could pass
        - a dataset constructor that generates the dataset for training from the original
        dataset (either by modifying the labels, the data, or removing some samples);
        - a metric that generates the final score: it could be either a Metric object from our library, or maybe
        accuracy comparison.

        :param device:
        :param args:
        :param kwargs:
        """
        self.device = device

    @abstractmethod
    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        *args,
        **kwargs,
    ):
        """
        Used to update the metric with new data.
        """

        raise NotImplementedError
