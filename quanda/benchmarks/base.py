"""Base class for all benchmarks."""

import os
import copy
import warnings
import zipfile
from abc import ABC
from typing import Callable, List, Optional, Union, Any

import requests
import torch
from datasets import load_dataset  # type: ignore
from tqdm import tqdm

from quanda.benchmarks.resources import (
    sample_transforms,
)
from quanda.benchmarks.resources.modules import (
    load_module_from_cfg,
)
from quanda.explainers import Explainer
from quanda.metrics import Metric
from quanda.utils.common import (
    get_load_state_dict_func,
    load_last_checkpoint,
    TrainValTest,
)
from quanda.utils.datasets.image_datasets import (
    HFtoTV,
    SingleClassImageDataset,
)
from quanda.utils.datasets.transformed import transform_wrappers

from quanda.utils.training import Trainer
from quanda.utils.training.options import optimizers, criteria, schedulers


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
        self.model: torch.nn.Module
        self.device: str
        self.train_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.checkpoints: Optional[List[str]]
        self.checkpoints_load_func: Optional[Callable[..., Any]]

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
        pass

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

    @classmethod
    def train(
        cls,
        config: dict,
        load_meta_from_disk: bool = True,
        device: str = "cpu",
        batch_size: int = 8,
    ):
        """Train a model using the provided configuration.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.
        load_meta_from_disk : bool, optional
            Whether to load metadata from disk, by default True
        device : str, optional
            Device to use for training, by default "cpu"
        batch_size : int, optional
            Batch size for training, by default 8

        Returns
        -------
        None
        """
        obj = cls.from_config(config, load_meta_from_disk=False, device=device)
        train_dataset, val_dataset, test_dataset = obj.split_dataset(
            obj.train_dataset, config["split_path"]
        )

        train_dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size
        )
        if val_dataset:
            val_dl = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size
            )
        else:
            val_dl = None

        obj.model.train()

        # Parse trainer configuration
        trainer_cfg = config["trainer"]

        # Get optimizer
        optimizer = optimizers[trainer_cfg["optimizer"]]

        # Get criterion
        criterion = criteria[trainer_cfg["criterion"]]()

        # Get scheduler if specified
        scheduler = None
        if trainer_cfg.get("scheduler"):
            scheduler = schedulers[trainer_cfg["scheduler"]]

        # Extract other parameters
        trainer_kwargs = {
            "optimizer": optimizer,
            "lr": trainer_cfg["lr"],
            "max_epochs": trainer_cfg["max_epochs"],
            "criterion": criterion,
            "scheduler": scheduler,
            "optimizer_kwargs": trainer_cfg.get("optimizer_kwargs", {}),
            "scheduler_kwargs": trainer_cfg.get("scheduler_kwargs", {}),
            "seed": trainer_cfg.get("seed", 42),
        }

        trainer = Trainer(**trainer_kwargs)

        trainer.fit(
            model=obj.model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
        )

        ckpt_dir = config["model"]["ckpt_dir"]
        torch.save(
            obj.model.state_dict(),
            ckpt_dir,
        )

        obj.model.to(obj.device)
        obj.model.eval()

        obj.save_metadata()

    def save_metadata(self):
        raise NotImplementedError

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

        Raises
        ------
        RuntimeError
            If downloading or extracting the zip file fails.
        """
        if os.path.exists(download_dir):
            warnings.warn(
                f"Directory {download_dir} already exists. Skipping download."
            )
            return download_dir

        os.makedirs(download_dir, exist_ok=True)
        zip_path = os.path.join(download_dir, "downloaded_file.zip")

        if not os.path.exists(zip_path):
            self._download_file(url, zip_path)

        self._extract_zip(zip_path, download_dir)
        return download_dir

    def _download_file(self, url: str, save_path: str):
        """Download file from URL with progress tracking.

        Parameters
        ----------
        url: str
            URL to download from.
        save_path: str
            Path to save the downloaded file.

        Raises
        ------
        RuntimeError
            If downloading fails.
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to download the zip file: {e}")

    def _extract_zip(self, zip_path: str, extract_dir: str):
        """Extract zip file to specified directory.

        Parameters
        ----------
        zip_path: str
            Path to the zip file.
        extract_dir: str
            Directory to extract to.

        Raises
        ------
        RuntimeError
            If extraction fails.
        """
        try:
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Failed to extract the zip file: {e}")

    def load_dataset_from_config(
        self, config: dict, load_meta_from_disk: bool = True
    ) -> torch.utils.data.Dataset:
        """Load dataset based on configuration."""
        if "dataset_str" in config:
            return self.load_hf_dataset(config, load_meta_from_disk)
        elif "zip_url" in config:
            return self.load_zip_dataset(config)
        return None

    def load_hf_dataset(
        self, config: dict, load_meta_from_disk: bool = True
    ) -> torch.utils.data.Dataset:
        """Load a HuggingFace dataset based on configuration."""
        transform = self.get_transform(config)
        base_dataset = self._process_dataset(
            dataset=config["dataset_str"],
            transform=transform,
            dataset_split=config.get("dataset_split", "train"),
        )
        return self.apply_indices(base_dataset, config, load_meta_from_disk)

    def load_zip_dataset(self, config: dict) -> torch.utils.data.Dataset:
        """Load a dataset from a zip file based on configuration."""
        transform = self.get_transform(config)
        download_dir = self.download_zip_file(
            url=config["zip_url"], download_dir=config["dataset_dir"]
        )
        return SingleClassImageDataset(
            root=download_dir,
            label=config["label"],
            transform=transform,
        )

    def get_transform(self, config: dict) -> Optional[Callable]:
        """Get the transform function from configuration."""
        return sample_transforms.get(config.get("transforms", None), None)

    def apply_indices(
        self,
        base_dataset: torch.utils.data.Dataset,
        config: dict,
        load_meta_from_disk: bool = True,
    ) -> torch.utils.data.Dataset:
        """Apply indices to the dataset based on configuration."""
        indices = copy.deepcopy(config.get("indices", "all"))
        if indices == "all":
            return base_dataset

        split_name = indices.pop("split_name", "train")
        split_filename = indices.pop("split_filename", "DOESNT_EXIST")
        metadata_dir = config.get("metadata_dir", ".tmp")
        split = self.load_split_if_exists_or_generate(
            base_dataset, load_meta_from_disk, metadata_dir, split_filename
        )
        return torch.utils.data.Subset(base_dataset, split[split_name])

    def dataset_from_cfg(
        self,
        config: dict,
        metadata_dir: str = ".tmp",
        load_meta_from_disk: bool = True,
    ):
        """Return the dataset using the given parameters."""
        if config is None:
            return None

        dataset = self.load_dataset_from_config(config)

        wrapper = copy.deepcopy(config.get("wrapper", None))
        if wrapper is not None:
            return self.apply_wrapper(
                dataset, wrapper, metadata_dir, load_meta_from_disk
            )
        return dataset

    def apply_wrapper(
        self,
        dataset: torch.utils.data.Dataset,
        wrapper: dict,
        metadata_dir: str,
        load_meta_from_disk: bool,
    ) -> torch.utils.data.Dataset:
        """Apply a wrapper to the dataset based on configuration."""
        wrapper_cls = transform_wrappers.get(wrapper.pop("type"))
        kwargs = wrapper
        if "metadata" in kwargs:
            metadata_args = kwargs.pop("metadata", {})
            meta_filename = metadata_args.pop(
                "metadata_filename", "DOESNT_EXIST"
            )
            if (
                wrapper_cls.metadata_cls.exists(metadata_dir, meta_filename)
                and load_meta_from_disk
            ):
                kwargs["metadata"] = wrapper_cls.metadata_cls.load(
                    metadata_dir, meta_filename
                )
            else:
                kwargs["metadata"] = wrapper_cls.metadata_cls(**metadata_args)

        if "sample_fn" in kwargs:
            kwargs["sample_fn"] = sample_transforms.get(kwargs["sample_fn"])
        if "dataset_transform" in kwargs:
            kwargs["dataset_transform"] = sample_transforms.get(
                kwargs["dataset_transform"]
            )
        dataset = wrapper_cls(dataset, **kwargs)
        dataset.metadata.save(
            metadata_dir, meta_filename
        )  # TODO: when to save metadata
        return dataset

    def split_dataset(
        self,
        dataset: torch.utils.data.Dataset,
        metadata_dir: str,
        split_filename: str,
        load_meta_from_disk: bool = True,
    ):
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

        split = self.load_split_if_exists_or_generate(
            dataset, load_meta_from_disk, metadata_dir, split_filename
        )

        train_dataset = torch.utils.data.Subset(dataset, split.train)
        test_dataset = torch.utils.data.Subset(dataset, split.test)

        if len(split.val) > 0:
            val_dataset = torch.utils.data.Subset(dataset, split.val)
        else:
            val_dataset = None

        return train_dataset, val_dataset, test_dataset

    def load_split_if_exists_or_generate(
        self, dataset, load_meta_from_disk, metadata_dir, split_filename
    ):
        if (
            TrainValTest.exists(metadata_dir, split_filename)
            and load_meta_from_disk
        ):
            split = TrainValTest.load(metadata_dir, split_filename)
        else:
            split = TrainValTest.split(len(dataset), 42, 0.1, 0.1)
            split.save(metadata_dir, split_filename)
        return split

    @classmethod
    def from_config(
        cls,
        config: dict,
        load_meta_from_disk: bool = True,
        device: str = "cpu",
    ):
        """Initialize the benchmark from a dictionary."""
        obj = cls()
        obj.device = device
        obj.train_dataset = obj.dataset_from_cfg(
            config=config.get("train_dataset"),
            metadata_dir=config.get("metadata_dir"),
            load_meta_from_disk=load_meta_from_disk,
        )
        obj.val_dataset = obj.dataset_from_cfg(
            config=config.get("val_dataset", None),
            metadata_dir=config.get("metadata_dir"),
            load_meta_from_disk=load_meta_from_disk,
        )
        obj.eval_dataset = obj.dataset_from_cfg(
            config=config.get("eval_dataset"),
            metadata_dir=config.get("metadata_dir"),
            load_meta_from_disk=load_meta_from_disk,
        )

        obj.model, obj.checkpoints = obj.model_from_cfg(config=config["model"])
        obj.checkpoints_load_func = None  # TODO: be more flexible
        return obj

    def model_from_cfg(self, config: dict) -> torch.nn.Module:
        """Return the model using the given parameters.

        Parameters
        ----------
        config : dict
            Dictionary containing the model configuration.

        Returns
        -------
        torch.nn.Module
            The model.

        """
        return load_module_from_cfg(config, self.device)
