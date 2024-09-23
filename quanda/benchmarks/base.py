import os
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union

import requests
import torch
from datasets import load_dataset  # type: ignore

from quanda.benchmarks.resources import benchmark_urls
from quanda.utils.datasets.image_datasets import HFtoTV


class Benchmark(ABC):
    """
    Base class for all benchmarks.

    Attributes:
        - name: str: The name of the benchmark.

    """

    name: str

    def __init__(self, *args, **kwargs):
        self.device: Optional[Union[str, torch.device]]
        self.bench_state: dict
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
    def download(cls, name: str, cache_dir: str, device: str, *args, **kwargs):
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """

        raise NotImplementedError

    def _get_bench_state(self, name: str, cache_dir: str, device: str, *args, **kwargs):
        # check if file exists
        if not os.path.exists(os.path.join(cache_dir, name + ".pth")):
            url = benchmark_urls[name]
            os.makedirs(os.path.join(cache_dir, name), exist_ok=True)

            # _get_bench_state to cache_dir
            response = requests.get(url)

            with open(os.path.join(cache_dir, name + ".pth"), "wb") as f:
                f.write(response.content)

        return torch.load(os.path.join(cache_dir, name + ".pth"), map_location=device)

    @classmethod
    @abstractmethod
    def assemble(cls, *args, **kwargs):
        """
        This method should assemble the benchmark components from arguments and persist them in the instance.
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
    ):
        """
        This method should set the device for the model.
        """
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

    def process_dataset(
        cls,
        train_dataset: Union[str, torch.utils.data.Dataset],
        transform: Optional[Callable] = None,
        dataset_split: str = "train",
    ):
        if isinstance(train_dataset, str):
            cls.dataset_str = train_dataset
            return HFtoTV(load_dataset(train_dataset, split=dataset_split), transform=transform)
        else:
            return train_dataset

    def build_eval_dataset(
        self,
        dataset_str: str,
        eval_indices: List[int],
        transform: Optional[Callable] = None,
        dataset_split: str = "test",
    ):
        test_dataset = HFtoTV(load_dataset(dataset_str, split=dataset_split), transform=transform)
        return torch.utils.data.Subset(test_dataset, eval_indices)
