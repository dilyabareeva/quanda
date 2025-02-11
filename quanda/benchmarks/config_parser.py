"""Configuration parser for benchmarks."""

import os
import copy
from typing import Optional, Tuple, Any, List, Union, Callable
import requests
import zipfile
import warnings
import torch
from datasets import load_dataset  # type: ignore
from quanda.utils.datasets.image_datasets import HFtoTV

from quanda.benchmarks.resources import (
    sample_transforms,
    pl_modules,
)
from quanda.utils.common import TrainValTest
from quanda.utils.datasets.image_datasets import (
    SingleClassImageDataset,
)
from quanda.utils.datasets.transformed import (
    transform_wrappers,
    TransformedDataset,
)
from quanda.utils.training import Trainer
from quanda.utils.training.options import optimizers, criteria, schedulers


class BenchConfigParser:
    """Parser for benchmark configurations."""

    @classmethod
    def parse_dataset_cfg(
        cls,
        ds_config: Optional[dict],
        metadata_dir: str = ".tmp",
        dataset_dir: str = ".tmp",
        load_meta_from_disk: bool = True,
    ):
        """Return the dataset using the given parameters."""
        if ds_config is None:
            return None

        dataset = cls._load_dataset_from_cfg(
            ds_config, dataset_dir, load_meta_from_disk
        )

        wrapper = copy.deepcopy(ds_config.get("wrapper", None))
        if wrapper is not None:
            return cls._apply_wrapper(
                dataset, wrapper, metadata_dir, load_meta_from_disk
            )
        return dataset

    @classmethod
    def parse_model_cfg(
        cls, model_cfg: dict, checkpoint_path: str, cfg_id: str
    ) -> Tuple[torch.nn.Module, List[str]]:
        """Parse model configuration and return the model and checkpoints.

        Parameters
        ----------
        model_cfg : dict
            Model configuration dictionary
        checkpoint_path : str
            Path to checkpoint directory
        cfg_id : str
            Configuration ID

        Returns
        -------
        Tuple[torch.nn.Module, List[str]]
            The configured model and list of checkpoints

        """
        module_cfg = model_cfg["module"]
        module = pl_modules[module_cfg["name"]](**module_cfg["args"])

        ckpt_dir = cls.get_ckpt_folder(model_cfg, checkpoint_path, cfg_id)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        checkpoints = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)]

        return module, checkpoints

    @classmethod
    def parse_trainer_cfg(cls, trainer_cfg: dict) -> Trainer:
        """Parse trainer configuration.

        Parameters
        ----------
        trainer_cfg : dict
            Trainer configuration dictionary

        Returns
        -------
        Dict[str, Any]
            Dictionary containing trainer configuration

        """
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

        return Trainer(**trainer_kwargs)

    @classmethod
    def _load_dataset_from_cfg(
        cls,
        ds_config: dict,
        dataset_dir: str,
        load_meta_from_disk: bool = True,
    ) -> torch.utils.data.Dataset:
        """Load dataset based on configuration."""
        if "dataset_str" in ds_config:
            return cls._load_hf_dataset(ds_config, load_meta_from_disk)
        elif "zip_url" in ds_config:
            return cls._load_zip_dataset(ds_config, dataset_dir=dataset_dir)
        else:
            raise ValueError("Dataset configuration not recognized.")

    @classmethod
    def _load_hf_dataset(
        cls, ds_config: dict, load_meta_from_disk: bool = True
    ) -> torch.utils.data.Dataset:
        """Load a HuggingFace dataset based on configuration."""
        transform = cls._get_transform(ds_config)
        base_dataset = cls.process_dataset(
            dataset=ds_config["dataset_str"],
            transform=transform,
            dataset_split=ds_config.get("dataset_split", "train"),
        )
        return cls._apply_indices(base_dataset, ds_config, load_meta_from_disk)

    @classmethod
    def _load_zip_dataset(
        cls, ds_config: dict, dataset_dir: str
    ) -> torch.utils.data.Dataset:
        """Load a dataset from a zip file based on configuration."""
        transform = cls._get_transform(ds_config)
        download_dir = cls._download_zip_file(
            url=ds_config["zip_url"], download_dir=dataset_dir
        )
        return SingleClassImageDataset(
            root=download_dir,
            label=ds_config["label"],
            transform=transform,
        )

    @classmethod
    def _get_transform(cls, ds_config: dict) -> Optional[Any]:
        """Get the transform function from configuration."""
        return sample_transforms.get(ds_config.get("transforms", None), None)

    @classmethod
    def _apply_indices(
        cls,
        base_dataset: torch.utils.data.Dataset,
        ds_config: dict,
        load_meta_from_disk: bool = True,
    ) -> torch.utils.data.Dataset:
        """Apply indices to the dataset based on configuration."""
        indices = copy.deepcopy(ds_config.get("indices", "all"))
        if indices == "all":
            return base_dataset

        split_name = indices.get("split_name", "train")
        split_filename = indices.get("split_filename", "DOESNT_EXIST")
        metadata_dir = ds_config.get("metadata_dir", ".tmp")
        split = cls._load_split_if_exists_or_generate(
            base_dataset, load_meta_from_disk, metadata_dir, split_filename
        )
        return torch.utils.data.Subset(base_dataset, split[split_name])

    @classmethod
    def _apply_wrapper(
        cls,
        dataset: torch.utils.data.Dataset,
        wrapper_cfg: dict,
        metadata_dir: str,
        load_meta_from_disk: bool,
    ) -> torch.utils.data.Dataset:
        """Apply a wrapper to the dataset based on configuration."""
        wrapper_cls = transform_wrappers[wrapper_cfg.pop("type")]
        # check if wrapper_cls is a subclass of TransformedDataset
        if not hasattr(wrapper_cls, "metadata_cls"):
            raise ValueError(
                "The wrapper class must be a subclass of TransformedDataset."
            )

        kwargs = wrapper_cfg
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
        wrapped_dataset: TransformedDataset = wrapper_cls(dataset, **kwargs)
        wrapped_dataset.metadata.save(
            metadata_dir, meta_filename
        )  # TODO: when to save metadata
        return wrapped_dataset

    @classmethod
    def _load_split_if_exists_or_generate(
        cls, dataset, load_meta_from_disk, metadata_dir, split_filename
    ):
        """Load the split if it exists, otherwise generate it."""
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
    def get_ckpt_folder(
        cls, model_cfg: dict, checkpoint_path: str, cfg_id: str
    ) -> str:
        """Get the checkpoint folder path."""
        ckpt_postfix = model_cfg.get("ckpt_postfix", "")
        return os.path.join(checkpoint_path, f"{cfg_id}_{ckpt_postfix}")

    @classmethod
    def process_dataset(
        cls,
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

    @classmethod
    def _download_zip_file(cls, url: str, download_dir: str) -> str:
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
            cls._download_file(url, zip_path)

        cls._extract_zip(zip_path, download_dir)
        return download_dir

    @classmethod
    def _download_file(cls, url: str, save_path: str):
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

    @classmethod
    def _extract_zip(cls, zip_path: str, extract_dir: str):
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

    @classmethod
    def split_dataset(
        cls,
        dataset: torch.utils.data.Dataset,
        metadata_dir: str,
        split_filename: str,
        load_meta_from_disk: bool = True,
    ):
        """Split the dataset using the given parameters.

        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            The dataset to be split.
        metadata_dir: str
            Directory to store the metadata.
        split_filename: str
            Name of the file to store the split.
        load_meta_from_disk: bool
            Whether to load metadata from disk.

        Returns
        -------
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset,
        torch.utils.data.Dataset]
            The train, val, test datasets.

        """
        split = cls._load_split_if_exists_or_generate(
            dataset, load_meta_from_disk, metadata_dir, split_filename
        )

        train_dataset = torch.utils.data.Subset(dataset, split.train)
        test_dataset = torch.utils.data.Subset(dataset, split.test)

        if len(split.val) > 0:
            val_dataset = torch.utils.data.Subset(dataset, split.val)
        else:
            val_dataset = None

        return train_dataset, val_dataset, test_dataset
