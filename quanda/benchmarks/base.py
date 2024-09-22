import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union

import requests
import torch
from datasets import load_dataset  # type: ignore
from tqdm import tqdm

from quanda.benchmarks.resources import benchmark_urls


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
    def download(
        cls, name: str, cache_dir: str, eval_dataset: torch.utils.data.Dataset, batch_size: int = 32, *args, **kwargs
    ):
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """

        # check if file exists
        if os.path.exists(os.path.join(cache_dir, name + ".pth")):
            return torch.load(os.path.join(cache_dir, name + ".pth"))

        url = benchmark_urls[name]
        os.makedirs(os.path.join(cache_dir, name), exist_ok=True)

        # download to cache_dir
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors

        # Get the total size of the content for the progress bar
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        # Initialize a bytes object to store the downloaded content
        content = bytes()

        # Progress bar setup
        with tqdm(total=total_size, unit="iB", unit_scale=True, unit_divisor=1024) as bar:
            for data in response.iter_content(block_size):
                content += data
                bar.update(len(data))

        torch.save(content, os.path.join(cache_dir, name + ".pth"))

        return content

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
        cls, train_dataset: Union[str, torch.utils.data.Dataset], dataset_split: str = "train", *args, **kwargs
    ):
        if isinstance(train_dataset, str):
            cls.dataset_str = train_dataset
            return load_dataset(train_dataset, split=dataset_split)
        else:
            return train_dataset

    def build_eval_dataset(self, dataset_str: str, eval_indices: List[int], dataset_split: str = "test", *args, **kwargs):
        test_dataset = load_dataset(dataset_str, split=dataset_split)
        return torch.utils.data.Subset(test_dataset, eval_indices)
