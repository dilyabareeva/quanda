from abc import ABC, abstractmethod
from typing import Optional, Union
import torch


class ToyBenchmark(ABC):
    def __init__(self, *args, **kwargs):
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
        self.model_device: Optional[Union[str, torch.device]]
        self.device: Optional[Union[str, torch.device]]

    @classmethod
    @abstractmethod
    def generate(cls, *args, **kwargs):
        """
        This method should generate all the benchmark components and persist them in the instance.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str, *args, **kwargs):
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def assemble(cls, *args, **kwargs):
        """
        This method should assemble the benchmark components from arguments and persist them in the instance.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, *args, **kwargs):
        """
        This method should save the benchmark components to a file/folder.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        *args,
        **kwargs,
    ):
        """
        Used to update the metric with new data.
        """

        raise NotImplementedError

    def set_devices(
        self,
        model: torch.nn.Module,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        This method should set the device for the model.
        """
        if next(model.parameters(), None) is not None:
            self.model_device = next(model.parameters()).device
        else:
            self.model_device = torch.device("cpu")

        self.device = device or self.model_device
