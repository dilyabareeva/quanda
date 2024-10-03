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
    """

    name: str

    def __init__(self, *args, **kwargs):
        self.device: Optional[Union[str, torch.device]]
        self.bench_state: dict
        self.dataset_str: Optional[str] = None
        self._checkpoint_paths: Optional[List[str]] = None

    @classmethod
    @abstractmethod
    def generate(cls, *args, **kwargs):
        """
        Generates the benchmark by specifying parameters.

        The evaluation can then be run using the `evaluate` method.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def download(cls, name: str, cache_dir: str, device: str, *args, **kwargs):
        """
        This method loads precomputed benchmark components from a file and creates an instance from the state dictionary.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        eval_dataset : torch.utils.data.Dataset
            Dataset to be used for the evaluation.
        batch_size : int, optional
            Batch size to be used, by default 32.

        Raises
        ------
        NotImplementedError
        """

        raise NotImplementedError

    def _get_bench_state(self, name: str, cache_dir: str, device: str, *args, **kwargs):
        """
        Downloads a benchmark state dictionary of a benchmark and returns.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        cache_dir : str
            Directory to store the downloaded benchmark components
        device : str
            Device to use with the benchmark components.

        Returns
        -------
        dict
            Benchmark state dictionary.
        """
        # check if file exists
        if not os.path.exists(os.path.join(cache_dir, name + ".pth")):
            url = benchmark_urls[name]
            os.makedirs(os.path.join(cache_dir, name), exist_ok=True)

            # download to cache_dir
            response = requests.get(url)

            with open(os.path.join(cache_dir, name + ".pth"), "wb") as f:
                f.write(response.content)

        return torch.load(os.path.join(cache_dir, name + ".pth"), map_location=device, weights_only=True)

    @classmethod
    @abstractmethod
    def assemble(cls, *args, **kwargs):
        """
        Assembles the benchmark from existing components.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        *args,
        **kwargs,
    ):
        """
        Run the evaluation using the benchmark.

        Raises
        ------
        NotImplementedError
        """

        raise NotImplementedError

    def set_devices(
        self,
        model: torch.nn.Module,
    ):
        """
        Infer device from model.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        """
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

    def process_dataset(
        cls,
        dataset: Union[str, torch.utils.data.Dataset],
        transform: Optional[Callable] = None,
        dataset_split: str = "train",
    ):
        """
        Return the dataset using the given parameters.

        Parameters
        ----------
        dataset : Union[str, torch.utils.data.Dataset]
            The dataset to be processed.
        transform : Optional[Callable], optional
            The transform to be applied to the dataset, by default None.
        dataset_split : str, optional
            The dataset split, by default "train", only used for HuggingFace datasets.

        Returns
        -------
        torch,utils.data.Dataset
            The dataset.
        """
        if isinstance(dataset, str):
            cls.dataset_str = dataset
            return HFtoTV(load_dataset(dataset, split=dataset_split), transform=transform)
        else:
            return dataset

    def build_eval_dataset(
        self,
        dataset_str: str,
        eval_indices: List[int],
        transform: Optional[Callable] = None,
        dataset_split: str = "test",
    ):
        """
        Downloads the HuggingFace evaluation dataset from given name.

        Parameters
        ----------
        dataset_str : str
            The name of the HuggingFace dataset.
        eval_indices : List[int]
            The indices to be used for evaluation.
        transform : Optional[Callable], optional
            The transform to be applied to the dataset, by default None.
        dataset_split : str, optional
            The dataset split, by default "test".

        Returns
        -------
        torch.utils.data.Dataset
            The evaluation dataset.
        """
        test_dataset = HFtoTV(load_dataset(dataset_str, split=dataset_split), transform=transform)
        return torch.utils.data.Subset(test_dataset, eval_indices)

    @property
    def checkpoint_paths(self) -> List[str]:
        assert (
            self._checkpoint_paths is not None
        ), "checkpoint_paths can only be called after instantiating a benchmark using the download method."
        return self._checkpoint_paths
