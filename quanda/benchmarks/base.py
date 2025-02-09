"""Base class for all benchmarks."""

import os
import warnings
import zipfile
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Union, Any, Dict

import requests
import torch
from datasets import load_dataset  # type: ignore
from tqdm import tqdm
import lightning as L

from quanda.benchmarks.resources import (
    benchmark_urls,
    sample_transforms,
    load_module_from_bench_state,
)
from quanda.benchmarks.resources.modules import bench_load_state_dict, \
    load_module_from_cfg
from quanda.explainers import Explainer
from quanda.metrics import Metric
from quanda.utils.common import get_load_state_dict_func, load_last_checkpoint, \
    TrainValTest
from quanda.utils.datasets.image_datasets import (
    HFtoTV,
    SingleClassImageDataset,
)
from quanda.utils.datasets.transformed import transform_wrappers

from quanda.utils.training import BaseTrainer


def process_dataset(
    dataset: Union[str, torch.utils.data.Dataset],
    transform: Optional[Callable] = None,
    dataset_split: str = "train",
    cache_dir: Optional[str] = None,
) -> torch.utils.data.Dataset:
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
    cache_dir : Optional[str], optional
        The cache directory, by default "~/.cache/huggingface/datasets".

    Returns
    -------
    torch,utils.data.Dataset
        The dataset.

    """
    if isinstance(dataset, str):

        if cache_dir is None:
            cache_dir = os.getenv(
                "HF_HOME",
                os.path.expanduser("~/.cache/huggingface/datasets"),
            )

        hf_dataset = load_dataset(
            "ylecun/mnist" if dataset == "mnist" else dataset,
            split=dataset_split,
            cache_dir=cache_dir,
        )
        return HFtoTV(hf_dataset, transform=transform)
    else:

        return dataset


class Benchmark(ABC):
    """Base class for all benchmarks."""

    name: str
    eval_args: List = []

    def __init__(self, *args, **kwargs):
        """Initialize the base `Benchmark` class."""
        self.device: str
        self.bench_state: dict
        self._checkpoints_load_func: Optional[Callable[..., Any]] = None
        self._checkpoints: Optional[Union[str, List[str]]] = None

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

        return obj._parse_bench_state(bench_state, cache_dir, device=device)

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
    
    def assemble(cls, *args, **kwargs):
        """Assembles the benchmark from existing components.

        Raises
        ------
        NotImplementedError

        """
        pass

    def _assemble_common(
        self,
        model: torch.nn.Module,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
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
        pass

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
        self,
        dataset: Union[str, torch.utils.data.Dataset],
        transform: Optional[Callable] = None,
        dataset_split: str = "train",
        cache_dir: Optional[str] = None,
    ) -> torch.utils.data.Dataset:
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
        cache_dir : Optional[str], optional
            The cache directory, by default "~/.cache/huggingface/datasets".

        Returns
        -------
        torch,utils.data.Dataset
            The dataset.

        """
        if isinstance(dataset, str):
            self.dataset_str = dataset
        return process_dataset(dataset, transform, dataset_split, cache_dir)

    def _build_eval_dataset(
        self,
        dataset_str: str,
        eval_indices: List[int],
        transform: Optional[Callable] = None,
        dataset_split: str = "test",
        cache_dir: Optional[str] = None,
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
        cache_dir : Optional[str], optional
            The HF dataset cache directory, by default None.

        Returns
        -------
        torch.utils.data.Dataset
            The evaluation dataset.

        """
        test_dataset = process_dataset(
            dataset=dataset_str,
            transform=transform,
            dataset_split=dataset_split,
            cache_dir=cache_dir,
        )
        return torch.utils.data.Subset(test_dataset, eval_indices)

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

    def _parse_bench_state(
        self,
        bench_state: dict,
        cache_dir: str,
        model_id: Optional[str] = None,
        device: str = "cpu",
    ):
        """Parse the benchmark state dictionary."""
        # TODO: this should be further refactored after the pipeline is done.
        # TODO: fix this mess.
        checkpoint_paths = []

        assemble_dict = {}

        for ckpt_name, ckpt in zip(
            bench_state["checkpoints"], bench_state["checkpoints_binary"]
        ):
            save_path = os.path.join(cache_dir, ckpt_name)
            torch.save(ckpt, save_path)
            checkpoint_paths.append(save_path)

        dataset_transform_str = bench_state.get("dataset_transform", None)
        dataset_transform = sample_transforms.get(dataset_transform_str, None)
        sample_fn_str = bench_state.get("sample_fn", None)
        sample_fn = sample_transforms.get(sample_fn_str, None)

        eval_dataset = self._build_eval_dataset(
            dataset_str=bench_state["dataset_str"],
            eval_indices=bench_state["eval_test_indices"],
            transform=dataset_transform
            if self.name != "Shortcut Detection"
            else None,  # TODO: better way to handle this
            dataset_split=bench_state.get("test_split_name", "test"),
        )

        if self.name == "Mixed Datasets":
            adversarial_dir_url = bench_state["adversarial_dir_url"]
            adversarial_dir = self.download_zip_file(
                url=adversarial_dir_url,
                download_dir=cache_dir,
            )

            adversarial_transform = sample_transforms[
                bench_state["adversarial_transform"]
            ]
            adv_test_indices = bench_state["adv_indices_test"]
            eval_from_test_indices = bench_state["eval_test_indices"]
            eval_indices = [
                adv_test_indices[i] for i in eval_from_test_indices
            ]

            eval_dataset = SingleClassImageDataset(
                root=adversarial_dir,
                label=bench_state["adversarial_label"],
                transform=adversarial_transform,
                indices=eval_indices,
            )

            adv_train_indices = bench_state["adv_indices_train"]
            assemble_dict["adversarial_dir"] = adversarial_dir
            assemble_dict["adv_train_indices"] = adv_train_indices
            assemble_dict["adversarial_transform"] = adversarial_transform

        module = load_module_from_bench_state(bench_state, device)

        # check the type of the instance self

        assemble_dict["model"] = module
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
            "adversarial_label",
            "global_method",
            "shortcut_indices",
            "shortcut_cls",
            "class_to_group",
        ]:
            if el in bench_state:
                assemble_dict[el] = bench_state[el]

        return self.assemble(**assemble_dict)

    def _evaluate_dataset(
        self,
        eval_dataset: torch.utils.data.Dataset,
        explainer: Explainer,
        metric: Metric,
        batch_size: int,
    ):
        expl_dl = torch.utils.data.DataLoader(
            eval_dataset, batch_size=batch_size
        )

        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (input, labels) in enumerate(pbar):
            pbar.set_description(
                "Metric evaluation, batch %d/%d" % (i + 1, n_batches)
            )

            input, labels = input.to(self.device), labels.to(self.device)

            if self.use_predictions:
                with torch.no_grad():
                    output = self.model(input)
                    targets = output.argmax(dim=-1)
            else:
                targets = labels

            explanations = explainer.explain(
                test_data=input,
                targets=targets,
            )
            data_unit = {
                "test_data": input,
                "test_targets": targets,
                "test_labels": labels,
                "explanations": explanations,
            }

            if self.name == "Subclass Detection":
                data_unit["grouped_labels"] = torch.tensor(
                    [self.class_to_group[i.item()] for i in labels],
                    device=labels.device,
                )
                if not self.use_predictions:
                    data_unit["targets"] = data_unit["grouped_labels"]

            eval_unit = {k: data_unit[k] for k in self.eval_args}
            metric.update(**eval_unit)

        return metric.compute()

    def _prepare_explainer(
        self,
        dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
    ):
        load_last_checkpoint(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
        )
        self.model.eval()

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            **expl_kwargs,
        )
        return explainer

    def _train_model(
        self,
        model: torch.nn.Module,
        trainer: Union[L.Trainer, BaseTrainer],
        train_dataset: torch.utils.data.Dataset,
        save_dir: str,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        train_dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size
        )
        if val_dataset:
            val_dl = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size
            )
        else:
            val_dl = None

        model.train()

        trainer_fit_kwargs = trainer_fit_kwargs or {}

        if isinstance(trainer, L.Trainer):
            if not isinstance(model, L.LightningModule):
                raise ValueError(
                    "Model should be a LightningModule if Trainer is a "
                    "Lightning Trainer"
                )

            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(model, torch.nn.Module):
                raise ValueError(
                    "Model should be a torch.nn.Module if Trainer is a "
                    "BaseTrainer"
                )

            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=val_dl,
                **trainer_fit_kwargs,
            )

        else:
            raise ValueError(
                "Trainer should be a Lightning Trainer or a BaseTrainer"
            )

        # save check point to cache_dir
        # TODO: add model id
        torch.save(
            model.state_dict(),
            save_dir,
        )

        model.to(self.device)
        model.eval()

        return model

    def download_zip_file(self, url: str, download_dir: str) -> str:
        """Download a zip file from the given URL and extract it.

        Parameters
        ----------
        url: str
            URL to the zip file.
        download_dir: str
            Path to the cache directory.

        Returns
        -------
        str
            Path to the extracted directory.

        """

        # if directory exists, return
        if os.path.exists(download_dir):
            warnings.warn(
                f"Directory {download_dir} already exists. Skipping download."
            )
            return download_dir

        os.makedirs(download_dir, exist_ok=True)

        zip_path = os.path.join(download_dir, "downloaded_file.zip")
        if not os.path.exists(zip_path):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download the zip file: {e}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(download_dir)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Failed to extract the zip file: {e}")

        return download_dir

    @staticmethod
    def dataset_from_cfg(config: dict, cache_dir: str):
        """Return the dataset using the given parameters.

        Parameters
        ----------
        config : dict
            Dictionary containing the dataset configuration.
        cache_dir : str
            The cache directory.

        Returns
        -------
        torch.utils.data.Dataset
            The dataset.

        """

        transform = sample_transforms.get(config.get("transforms", None), None)
        if "dataset_str" in config:
            base_dataset = process_dataset(
                dataset=config["dataset_str"],
                transform=transform,
                dataset_split=config.get("dataset_split", "train"),
            )

            wrapper = config.get("wrapper", None)

            if wrapper is not None:
                wrapper_cls = transform_wrappers.get(wrapper.pop("type"))
                kwargs = wrapper
                if "metadata" in kwargs:
                    metadata_args = kwargs.pop("metadata", {})
                    kwargs["metadata"] = wrapper_cls.metadata_cls(
                        **metadata_args)
                if "sample_fn" in kwargs:
                    kwargs["sample_fn"] = sample_transforms.get(
                        kwargs["sample_fn"])
                if "dataset_transform" in kwargs:
                    kwargs["dataset_transform"] = sample_transforms.get(
                        kwargs["dataset_transform"]
                    )
                dataset = wrapper_cls(base_dataset, **kwargs)
                return dataset
            else:
                return base_dataset

        elif "zip_url" in config:
            # TODO: make it jore
            download_dir = Benchmark().download_zip_file(
                url=config["zip_url"], download_dir=config["dataset_dir"]
            )
            return SingleClassImageDataset(
                root=download_dir,
                label=config["label"],
                transform=transform,
                indices=config.get("indices", None),
            )

    def model_from_cfg(self, config: dict, cache_dir: str):
        """Return the model using the given parameters.

        Parameters
        ----------
        config : dict
            Dictionary containing the model configuration.
        cache_dir : str
            The cache directory.

        Returns
        -------
        torch.nn.Module
            The model.

        """
        return load_module_from_cfg(config, self.device)

    def split_dataset_from_cfg(self, dataset: torch.utils.data.Dataset, split_path: str):
        """
        Split the dataset using the given parameters.

        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            The dataset to be split.
        split: str
            Path to file where config in TrainValTest format is stored.

        Returns
        -------
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, torch.utils.data.Dataset]
            The train, val, test datasets.
        """
        # split split_path to folder and file name
        folder, file = os.path.split(split_path)
        split = TrainValTest.load(folder, file)


        train_dataset = torch.utils.data.Subset(dataset, split.train)
        test_dataset = torch.utils.data.Subset(dataset, split.test)

        if split.val:
            val_dataset = torch.utils.data.Subset(dataset, split.val)
        else:
            val_dataset = None

        return train_dataset, val_dataset, test_dataset


