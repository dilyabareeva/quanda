"""Configuration parser for benchmarks."""

import copy
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import datasets as hf_datasets  # type: ignore
import torch
from datasets import load_dataset  # type: ignore
from huggingface_hub import snapshot_download

from quanda.benchmarks.resources import pl_modules
from quanda.benchmarks.resources.sample_transforms import sample_transforms
from quanda.utils.common import DatasetSplit, ds_len
from quanda.utils.datasets.image_datasets import HFtoTV
from quanda.utils.datasets.transformed import (
    TransformedDataset,
    transform_wrappers,
)
from quanda.utils.tokenization import tokenize_dataset
from quanda.utils.training import Trainer
from quanda.utils.training.options import criteria, optimizers, schedulers


class BenchConfigParser:
    """Parser for benchmark configurations."""

    @classmethod
    def get_metadata_dir(
        cls,
        cfg: dict,
        bench_save_dir: str = ".tmp",
        suffix: str = "",
    ):
        """Parse metadata configuration and return the metadata directory."""
        base_metadata_dir = os.path.join(bench_save_dir, "metadata")
        # create metadata_dir if it doesn't exist
        os.makedirs(base_metadata_dir, exist_ok=True)
        metadata_dir = os.path.join(
            base_metadata_dir, f"{cfg['id']}_metadata{suffix}"
        )

        # create metadata_dir if it doesn't exist
        os.makedirs(metadata_dir, exist_ok=True)

        return metadata_dir

    @classmethod
    def load_metadata(
        cls,
        cfg: dict,
        metadata_dir: str = ".tmp/meta",
    ):
        """Load metadata from the given configuration."""
        meta_id = cfg.get("meta_id", f"{cfg['repo_id']}/{cfg['id']}_metadata")

        return snapshot_download(
            repo_id=meta_id, local_dir=metadata_dir, repo_type="dataset"
        )

    @classmethod
    def parse_dataset_cfg(
        cls,
        ds_config: Optional[dict],
        metadata_dir: str = ".tmp/meta",
        load_meta_from_disk: bool = True,
        reuse_split: bool = False,
    ):
        """Return the dataset using the given parameters."""
        if ds_config is None:
            return None

        dataset = cls._load_dataset_from_cfg(
            ds_config, metadata_dir, load_meta_from_disk, reuse_split
        )

        wrapper = copy.deepcopy(ds_config.get("wrapper", None))
        if wrapper is not None:
            return cls._apply_wrapper(
                dataset, ds_config, wrapper, metadata_dir, load_meta_from_disk
            )
        return dataset

    @classmethod
    def parse_model_cfg(
        cls,
        model_cfg: dict,
        bench_save_dir: str,
        ckpts: List[str],
        load_model_from_disk: bool,
        device: str,
    ) -> Tuple[torch.nn.Module, List[str], Callable]:
        """Parse model configuration and return the model and checkpoints.

        Parameters
        ----------
        model_cfg : dict
            Model configuration dictionary
        bench_save_dir : str
            Path to checkpoint directory "ckpt".
        ckpts: List[str]
            File names of checkpoints.
        load_model_from_disk : bool
            If True, the method tries to load the model from the local cache.
        load_model_from_disk : bool
            If True, the method tries to load the model from the local cache.
        device : str
            Device to use for the model.

        Returns
        -------
        Tuple[torch.nn.Module, List[str]]
            The configured model and list of checkpoints

        """
        module_cfg = model_cfg["module"]
        module_cls = pl_modules[module_cfg["name"]]
        module = module_cls(**module_cfg["args"])

        checkpoint_path = os.path.join(bench_save_dir, "ckpt")
        ckpt_ids = [f"{ckpt}" for ckpt in ckpts]

        if not hasattr(module_cls, "from_pretrained"):
            raise ValueError(f"Model class {module_cls} is not HF compatible.")

        def load_state_dict(model: torch.nn.Module, ckpt_str: str):
            ckpt = ckpt_str.split("/")[-1]
            ckpt_dir = os.path.join(checkpoint_path, ckpt)
            if os.path.exists(os.path.join(ckpt_dir, "config.json")):
                pretrained_model_name_or_path = ckpt_dir
                cache_dir = None
            else:
                pretrained_model_name_or_path = ckpt_str
                cache_dir = ckpt_dir

            try:
                pretrained_model = module_cls.from_pretrained(
                    pretrained_model_name_or_path=pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    local_files_only=load_model_from_disk,
                )
            except Exception as e:
                raise ValueError(
                    f"Error loading model from "
                    f"{pretrained_model_name_or_path}: {e}"
                )
            model.load_state_dict(pretrained_model.state_dict())
            model.to(device)
            return model_cfg["trainer"]["lr"]

        # check if dir is empty
        return module, ckpt_ids, load_state_dict

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
            "num_workers": trainer_cfg.get("num_workers", 0),
            "enable_progress_bar": trainer_cfg.get(
                "enable_progress_bar", True
            ),
            "gradient_clip_val": trainer_cfg.get("gradient_clip_val", None),
        }

        return Trainer(**trainer_kwargs)

    @classmethod
    def _load_dataset_from_cfg(
        cls,
        ds_config: dict,
        metadata_dir: str,
        load_meta_from_disk: bool = True,
        reuse_split: bool = False,
    ) -> torch.utils.data.Dataset:
        """Load dataset based on configuration."""
        if "single_class_dataset" not in ds_config:
            return cls._load_hf_dataset_from_config(
                ds_config, metadata_dir, load_meta_from_disk, reuse_split
            )
        elif ds_config["single_class_dataset"]:
            return cls._load_single_class_dataset(
                ds_config,
            )
        else:
            raise ValueError("Dataset configuration not recognized.")

    @classmethod
    def _load_hf_dataset_from_config(
        cls,
        ds_config: dict,
        metadata_dir: str,
        load_meta_from_disk: bool = True,
        reuse_split: bool = False,
    ) -> Union[torch.utils.data.Dataset, hf_datasets.Dataset]:
        """Load a HuggingFace dataset based on configuration."""
        transform = cls._get_transform(ds_config)
        tokenizer_cfg = ds_config.get("tokenizer", None)
        base_dataset = cls._parse_hf_dataset(
            dataset=ds_config["dataset_str"],
            transform=transform,
            dataset_split=ds_config.get("dataset_split", "train"),
            tokenizer_cfg=tokenizer_cfg,
            dataset_config=ds_config.get("dataset_config", None),
        )
        load_split = load_meta_from_disk or reuse_split
        return cls._apply_indices(
            base_dataset, ds_config, metadata_dir, load_split
        )

    @classmethod
    def _load_single_class_dataset(
        cls, ds_config: dict, cache_dir: Optional[str] = None
    ) -> torch.utils.data.Dataset:
        """Load a dataset from a zip file based on configuration."""
        transform = cls._get_transform(ds_config)
        transform = transform if transform is not None else lambda x: x

        base_dataset = load_dataset(
            ds_config["dataset_str"],
            name=ds_config.get("dataset_config", None),
            split=ds_config.get("dataset_split", "train"),
            cache_dir=cache_dir,
        )
        if "label" in ds_config:
            return HFtoTV(
                base_dataset.map(
                    lambda x: {
                        "label": ds_config["label"],
                        "image": transform(x["image"]),
                    }
                ).with_format("torch"),
                transform=None,
            )
        else:
            return HFtoTV(base_dataset, transform=transform)

    @classmethod
    def _get_transform(cls, ds_config: dict) -> Optional[Any]:
        """Get the transform function from configuration."""
        transform_name = ds_config.get("transforms", None)
        return (
            sample_transforms.get(transform_name) if transform_name else None
        )

    @classmethod
    def _apply_indices(
        cls,
        base_dataset: Union[torch.utils.data.Dataset, hf_datasets.Dataset],
        ds_config: dict,
        metadata_dir: str,
        load_meta_from_disk: bool = True,
    ) -> Union[torch.utils.data.Dataset, hf_datasets.Dataset]:
        """Apply indices to the dataset based on configuration."""
        indices = copy.deepcopy(ds_config.get("indices", "all"))
        final_indices = list(range(ds_len(base_dataset)))
        if indices != "all":
            split_name = indices.get("split_name", "train")
            split_filename = indices.get("split_filename", "DOESNT_EXIST")
            split_ratios = indices.get(
                "split_ratios", {"train": 1.0, "test": 0.0, "val": 0.0}
            )
            split = cls._load_split_if_exists_or_generate(
                base_dataset,
                load_meta_from_disk,
                metadata_dir,
                split_filename,
                split_ratios=split_ratios,
            )
            final_indices = split[split_name]

        if isinstance(base_dataset, hf_datasets.Dataset):
            base_dataset = base_dataset.select(final_indices)
        else:
            base_dataset = torch.utils.data.Subset(base_dataset, final_indices)

        return base_dataset

    @classmethod
    def _apply_filter(
        cls,
        dataset: torch.utils.data.Dataset,
        ds_config: dict,
        metadata_dir: str,
        load_meta_from_disk: bool = True,
    ):
        """Apply the filter to the dataset and save the indices."""
        filter_indices_cfg = ds_config.get("filter_indices", None)
        if filter_indices_cfg is not None:
            filter_filename = filter_indices_cfg.get(
                "split_filename", "DOESNT_EXIST"
            )
            filter_split_name = filter_indices_cfg.get("split_name", "default")
            if (
                DatasetSplit.exists(metadata_dir, filter_filename)
                and load_meta_from_disk
            ):
                filter_split = DatasetSplit.load(metadata_dir, filter_filename)
                filter_indices = filter_split[filter_split_name].flatten()
                if isinstance(dataset, TransformedDataset):
                    dataset.apply_filter(filter_indices)
                else:
                    dataset = torch.utils.data.Subset(dataset, filter_indices)
                return dataset
        return dataset

    @classmethod
    def _apply_wrapper(
        cls,
        dataset: torch.utils.data.Dataset,
        ds_config: dict,
        wrapper_cfg: dict,
        metadata_dir: str,
        load_meta_from_disk: bool,
    ) -> torch.utils.data.Dataset:
        """Apply a wrapper to the dataset based on configuration."""
        wrapper_cfg = dict(wrapper_cfg)
        wrapper_cls = transform_wrappers[wrapper_cfg.pop("type")]
        # check if wrapper_cls is a subclass of TransformedDataset
        if not hasattr(wrapper_cls, "metadata_cls"):
            raise ValueError(
                "The wrapper class must be a subclass of TransformedDataset."
            )

        kwargs = wrapper_cfg
        if "metadata" in kwargs:
            metadata_args = dict(kwargs.pop("metadata", {}))
            meta_filename = metadata_args.pop(
                "metadata_filename", "DOESNT_EXIST"
            )
            if (
                wrapper_cls.metadata_cls.exists(metadata_dir, meta_filename)
                and load_meta_from_disk
            ):
                loaded_meta = wrapper_cls.metadata_cls.load(
                    metadata_dir, meta_filename
                )
                kwargs["metadata"] = loaded_meta
            else:
                kwargs["metadata"] = wrapper_cls.metadata_cls(**metadata_args)

        if "sample_fn" in kwargs:
            kwargs["sample_fn"] = sample_transforms.get(kwargs["sample_fn"])
        if "dataset_transform" in kwargs:
            kwargs["dataset_transform"] = sample_transforms.get(
                kwargs["dataset_transform"]
            )
        wrapped_dataset: TransformedDataset = wrapper_cls(dataset, **kwargs)
        filtered_dataset = cls._apply_filter(
            wrapped_dataset,
            ds_config,
            metadata_dir,
            load_meta_from_disk,
        )
        if not load_meta_from_disk:
            filtered_dataset.metadata.save(metadata_dir, meta_filename)
        return filtered_dataset

    @classmethod
    def _load_split_if_exists_or_generate(
        cls,
        dataset,
        load_meta_from_disk,
        metadata_dir,
        split_filename,
        split_ratios: Optional[dict] = None,
    ):
        """Load the split if it exists, otherwise generate it."""
        if split_ratios is None:
            split_ratios = {"train": 0.9, "test": 0.1}
        if (
            DatasetSplit.exists(metadata_dir, split_filename)
            and load_meta_from_disk
        ):
            split = DatasetSplit.load(metadata_dir, split_filename)
        else:
            split = DatasetSplit.split(len(dataset), 42, split_ratios)
            split.save(metadata_dir, split_filename)
        return split

    @classmethod
    def _parse_hf_dataset(
        cls,
        dataset: Union[str, torch.utils.data.Dataset],
        transform: Optional[Callable] = None,
        dataset_split: str = "train",
        cache_dir: Optional[str] = None,
        tokenizer_cfg: Optional[dict] = None,
        dataset_config: Optional[str] = None,
    ) -> Union[torch.utils.data.Dataset, hf_datasets.Dataset]:
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
        tokenizer_cfg : Optional[dict], optional
            Tokenizer configuration for text datasets. When provided,
            the dataset is tokenized and returned as an HF Dataset
            instead of being wrapped with HFtoTV.
        dataset_config : Optional[str], optional
            The dataset configuration/subset name, by default None.

        Returns
        -------
        Union[torch.utils.data.Dataset, datasets.Dataset]
            The dataset.

        """
        if isinstance(dataset, str):
            hf_dataset = load_dataset(
                "ylecun/mnist" if dataset == "mnist" else dataset,
                name=dataset_config,
                split=dataset_split,
                cache_dir=cache_dir,
            )
            if tokenizer_cfg is not None:
                return tokenize_dataset(hf_dataset, tokenizer_cfg)
            return HFtoTV(hf_dataset, transform=transform)
        else:
            return dataset

    @classmethod
    def split_dataset(
        cls,
        dataset: torch.utils.data.Dataset,
        ds_config: dict,
        metadata_dir: str,
        load_meta_from_disk: bool = True,
    ):
        """Split the dataset using the given parameters.

        Parameters
        ----------
        dataset: torch.utils.data.Dataset
            The dataset to be split.
        ds_config: dict
            The dataset configuration dictionary.
        metadata_dir: str
            Directory to store the metadata.
        load_meta_from_disk: bool
            Whether to load metadata from disk.

        Returns
        -------
        Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset,
        torch.utils.data.Dataset]
            The train, val, test datasets.

        """
        indices = ds_config.get("indices", "all")
        if indices == "all":
            return {"train": dataset, "val": None, "test": None}

        split_filename = indices.get("split_filename", "DOESNT_EXIST")
        split_ratios = indices.get(
            "split_ratios", {"train": 1.0, "test": 0.0, "val": 0.0}
        )
        splits = cls._load_split_if_exists_or_generate(
            dataset,
            load_meta_from_disk,
            metadata_dir,
            split_filename,
            split_ratios=split_ratios,
        ).splits

        return {
            k: torch.utils.data.Subset(dataset, splits[k])
            if len(splits[k]) > 0
            else None
            for k in splits.keys()
        }

    @classmethod
    def parse_logger(cls, cfg):
        """Parse the logger configuration from the main config.

        Parameters
        ----------
        cfg: DictConfig
            The main configuration dictionary.

        Returns
        -------
        Any
            The logger instance

        """
        # Import Hydra only when needed
        try:
            from hydra.utils import instantiate
        except ImportError:
            raise ImportError(
                "Hydra is not installed, but `instantiate` was requested. "
                "Either install Hydra (`pip install hydra-core`) or modify "
                "the config parsing."
            )

        logger_cfg = cfg.get("logger", None)
        logger = instantiate(logger_cfg)

        return logger
