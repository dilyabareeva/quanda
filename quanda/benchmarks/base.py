"""Base class for all benchmarks."""

import os
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union, Any

import requests
import torch
from datasets import load_dataset  # type: ignore

from quanda.benchmarks.resources import (
    benchmark_urls,
    sample_transforms,
    load_module_from_bench_state,
)
from quanda.benchmarks.resources.modules import bench_load_state_dict
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

        Returns
        -------
        Benchmark
            The benchmark instance.

        """
        obj = cls()
        bench_state = obj._get_bench_state(
            name, cache_dir, device, *args, **kwargs
        )

        return obj.parse_bench_state(bench_state, cache_dir, device=device)

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

    def _assemble_common(
        self,
        model: torch.nn.Module,
        eval_dataset: torch.utils.data.Dataset,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        use_predictions: bool = True,
    ):
        """Assembles the benchmark from existing components.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to generate attributions.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset to be used for the benchmark.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation, by
            default True.

        Returns
        -------
        None
        """
        self.model = model
        self._set_devices(model)
        self.eval_dataset = eval_dataset
        self.checkpoints = checkpoints
        self.checkpoints_load_func = checkpoints_load_func
        self.use_predictions = use_predictions

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

    def parse_bench_state(
        self,
        bench_state: dict,
        cache_dir: Optional[str] = None,
        model_id: Optional[str] = None,
        device: str = "cpu",
    ):
        """Parse the benchmark state dictionary."""

        checkpoint_paths = []

        assemble_dict = {}

        for ckpt_name, ckpt in zip(
            bench_state["checkpoints"], bench_state["checkpoints_binary"]
        ):
            save_path = os.path.join(cache_dir, ckpt_name)
            torch.save(ckpt, save_path)
            checkpoint_paths.append(save_path)

        dataset_transform_str = bench_state.get("dataset_transform", None)
        if dataset_transform_str:
            dataset_transform = sample_transforms[dataset_transform_str]
        else:
            dataset_transform = None

        sample_fn_str = bench_state.get("sample_fn", None)
        if sample_fn_str:
            sample_fn = sample_transforms[sample_fn_str]
        else:
            sample_fn = None

        eval_dataset = self._build_eval_dataset(
            dataset_str=bench_state["dataset_str"],
            eval_indices=bench_state["eval_test_indices"],
            transform=dataset_transform
            if self.name != "Shortcut Detection"
            else None,  # TODO: better way to handle this
            dataset_split=bench_state["test_split_name"],
        )

        module = load_module_from_bench_state(bench_state, device)

        # check the type of the instance self

        assemble_dict["model"] = module
        assemble_dict["group_model"] = (
            module  # TODO: rename model to group_model in the benchmarks
        )
        assemble_dict["checkpoints"] = bench_state["checkpoints_binary"]
        assemble_dict["checkpoints_load_func"] = bench_load_state_dict
        assemble_dict["train_dataset"] = bench_state["dataset_str"]
        assemble_dict["base_dataset"] = bench_state[
            "dataset_str"
        ]  # TODO: rename dataset_str to base/train_dataset_str
        assemble_dict["eval_dataset"] = eval_dataset
        assemble_dict["use_predictions"] = bench_state["use_predictions"]
        assemble_dict["checkpoint_paths"] = checkpoint_paths
        assemble_dict["dataset_transform"] = dataset_transform
        assemble_dict["sample_fn"] = sample_fn
        assemble_dict["cache_dir"] = cache_dir
        assemble_dict["model_id"] = model_id

        for el in [
            "n_classes",
            "mislabeling_labels",
            "global_method",
            "shortcut_indices",
            "shortcut_cls",
            "class_to_group",
        ]:
            if el in bench_state:
                assemble_dict[el] = bench_state[el]

        return self.assemble(**assemble_dict)
