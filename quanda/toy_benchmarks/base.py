from abc import ABC, abstractmethod
from typing import Optional, Union
from datasets import load_dataset

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
        self.device: Optional[Union[str, torch.device]]
        self.bench_state: dict = {}
        self.hf_dataset_bool: bool
        self.train_dataset: torch.utils.data.Dataset
        self.dataset_str: Optional[str] = None

    @classmethod
    @abstractmethod
    def generate(cls, *args, **kwargs):
        """
        This method should generate all the benchmark components and persist them in the instance.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def download(cls, name: str, *args, **kwargs):
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
    def save(self, path: str, *args, **kwargs):
        """
        This method should save the benchmark components to a file/folder.
        """
        if len(self.bench_state) == 0:
            raise ValueError("No benchmark components to save.")
        torch.save(self.bench_state, path)

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
    ):
        """
        This method should set the device for the model.
        """
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

    def set_dataset(cls, train_dataset: Optional[str, torch.utils.data.Dataset], *args, **kwargs):
        """
        This method should generate all the benchmark components and persist them in the instance.
        """
        if isinstance(train_dataset, str):
            cls.train_dataset = load_dataset(train_dataset)
            cls.hf_dataset_bool = True
            cls.dataset_str = train_dataset
        else:
            cls.train_dataset = train_dataset
            cls.hf_dataset_bool = False
