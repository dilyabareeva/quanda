"""Base class for all benchmarks."""

import os
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union, Any

import requests
import torch
from datasets import load_dataset  # type: ignore

from quanda.benchmarks.resources import benchmark_urls
from quanda.utils.common import get_load_state_dict_func
from quanda.utils.datasets.image_datasets import HFtoTV


class Benchmark(ABC):
    """Base class for all benchmarks."""

    name: str

    def __init__(self, *args, **kwargs):
        """Initialize the base `Benchmark` class."""
        self.device: Union[str, torch.device]
        self.bench_state: dict
        self._checkpoint_paths: Optional[List[str]] = None
        self._checkpoints_load_func: Optional[Callable[..., Any]] = None
        self._checkpoints: Optional[Union[str, List[str]]] = None

    @classmethod
    @abstractmethod
    def generate(cls, *args, **kwargs):
        """Generate the benchmark by specifying parameters.

        The evaluation can then be run using the `evaluate` method.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def download(cls, name: str, cache_dir: str, device: str, *args, **kwargs):
        """Download a precomputed benchmark.

        Load precomputed benchmark components from a file and creates an
        instance from the state dictionary.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        cache_dir : str
            Directory to store the downloaded benchmark components.
        device : str
            Device to load the model on.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def _get_bench_state(
        self,
        name: str,
        cache_dir: str,
        device: str,
    ):
        """Download a benchmark state dictionary of a benchmark and returns.

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
        os.makedirs(cache_dir, exist_ok=True)
        # check if file exists
        if not os.path.exists(os.path.join(cache_dir, name + ".pth")):
            url = benchmark_urls[name]

            # download to cache_dir
            response = requests.get(url)

            with open(os.path.join(cache_dir, name + ".pth"), "wb") as f:
                f.write(response.content)

        return torch.load(
            os.path.join(cache_dir, name + ".pth"),
            map_location=device,
            weights_only=True,
        )

    @classmethod
    @abstractmethod
    def assemble(cls, *args, **kwargs):
        """Assembles the benchmark from existing components.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        """Run the evaluation using the benchmark.

        Parameters
        ----------
        explainer_cls : type
            The explainer class to be used for evaluation.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments to be passed to the explainer, by
            default None.
        batch_size : int, optional
            Batch size for the evaluation, by default 8.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def _set_devices(
        self,
        model: torch.nn.Module,
    ):
        """Infer device from model.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.

        """
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

    def _process_dataset(
        cls,
        dataset: Union[str, torch.utils.data.Dataset],
        transform: Optional[Callable] = None,
        dataset_split: str = "train",
    ):
        """Return the dataset using the given parameters.

        Parameters
        ----------
        dataset : Union[str, torch.utils.data.Dataset]
            The dataset to be processed.
        transform : Optional[Callable], optional
            The transform to be applied to the dataset, by default None.
        dataset_split : str, optional
            The dataset split, by default "train", only used for HuggingFace
            datasets.

        Returns
        -------
        torch,utils.data.Dataset
            The dataset.

        """
        if isinstance(dataset, str):
            cls.dataset_str = dataset
            return HFtoTV(
                load_dataset(dataset, split=dataset_split), transform=transform
            )
        else:
            return dataset

    def _build_eval_dataset(
        self,
        dataset_str: str,
        eval_indices: List[int],
        transform: Optional[Callable] = None,
        dataset_split: str = "test",
    ):
        """Download the HuggingFace evaluation dataset from given name.

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
        test_dataset = HFtoTV(
            load_dataset(dataset_str, split=dataset_split), transform=transform
        )
        return torch.utils.data.Subset(test_dataset, eval_indices)

    def get_checkpoint_paths(self) -> List[str]:
        """Return the paths to the checkpoints."""
        assert self._checkpoint_paths is not None, (
            "get_checkpoint_paths can only be called after instantiating a "
            "benchmark using the download method."
        )
        return self._checkpoint_paths

    @property
    def checkpoints_load_func(self):
        """Return the function to load the checkpoints."""
        return self._checkpoints_load_func

    @checkpoints_load_func.setter
    def checkpoints_load_func(self, value):
        """Set the function to load the checkpoints."""
        if self.device is None:
            raise ValueError(
                "The device must be set before setting the "
                "checkpoints_load_func."
            )
        if value is None:
            self._checkpoints_load_func = get_load_state_dict_func(self.device)
        else:
            self._checkpoints_load_func = value

    @property
    def checkpoints(self):
        """Return the checkpoint paths."""
        return self._checkpoints

    @checkpoints.setter
    def checkpoints(self, value):
        """Set the checkpoint paths."""
        if value is None:
            self._checkpoints = []
        else:
            self._checkpoints = value if isinstance(value, List) else [value]
