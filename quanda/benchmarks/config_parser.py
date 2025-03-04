"""Configuration parser for benchmarks."""

import os
import copy
from typing import Optional, Tuple, Any, List, Union, Callable
from huggingface_hub import snapshot_download
import torch
from datasets import load_dataset  # type: ignore
from quanda.utils.datasets.image_datasets import HFtoTV

from quanda.benchmarks.resources import (
    sample_transforms,
    pl_modules,
)
from quanda.utils.common import TrainValTest
from quanda.utils.datasets.transformed import (
    transform_wrappers,
    TransformedDataset,
)
from quanda.utils.training import Trainer
from quanda.utils.training.options import optimizers, criteria, schedulers


class BenchConfigParser:
    """Parser for benchmark configurations."""

    @classmethod
    def load_metadata(
        cls,
        cfg: dict,
        bench_save_dir: str = ".tmp",
        load_meta_from_disk: bool = True,
    ):
        """Parse metadata configuration and return the metadata directory."""
        meta_id = cfg.get("meta_id", f"{cfg['id']}_metadata")
        repo_id = f"{cfg['repo_id']}/{meta_id}"
        base_metadata_dir = os.path.join(bench_save_dir, "metadata")
        # create metadata_dir if it doesn't exist
        os.makedirs(base_metadata_dir, exist_ok=True)
        metadata_dir = os.path.join(base_metadata_dir, f"{cfg['id']}_metadata")
        if os.path.exists(metadata_dir) or not load_meta_from_disk:
            return metadata_dir
        return snapshot_download(
            repo_id=repo_id, local_dir=metadata_dir, repo_type="dataset"
        )

    @classmethod
    def parse_dataset_cfg(
        cls,
        ds_config: Optional[dict],
        metadata_dir: str = ".tmp/meta",
        bench_save_dir: str = ".tmp",
        load_meta_from_disk: bool = True,
    ):
        """Return the dataset using the given parameters."""
        if ds_config is None:
            return None

        dataset = cls._load_dataset_from_cfg(
            ds_config, bench_save_dir, load_meta_from_disk
        )

        wrapper = copy.deepcopy(ds_config.get("wrapper", None))
        if wrapper is not None:
            return cls._apply_wrapper(
                dataset, wrapper, metadata_dir, load_meta_from_disk
            )
        return dataset

    @classmethod
    def parse_model_cfg(
        cls,
        model_cfg: dict,
        bench_save_dir: str,
        repo_id: str,
        cfg_id: str,
        offline: bool,
        device: str,
    ) -> Tuple[torch.nn.Module, List[str], Callable]:
        """Parse model configuration and return the model and checkpoints.

        Parameters
        ----------
        model_cfg : dict
            Model configuration dictionary
        bench_save_dir : str
            Path to checkpoint directory "ckpt".
        repo_id : str
            Repo ID Hugging Face
        cfg_id : str
            Configuration ID
        offline : bool
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
        ckpt_dir = cls.get_ckpt_folder(model_cfg, checkpoint_path, cfg_id)
        ckpt_id = f"{repo_id}/{cfg_id}"

        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        if not hasattr(module_cls, "from_pretrained"):
            raise ValueError(f"Model class {module_cls} is not HF compatible.")

        def load_state_dict(model: torch.nn.Module, ckpt_str: str):
            pretrained_model = module_cls.from_pretrained(
                ckpt_str,
                cache_dir=ckpt_dir,
                local_files_only=offline,
            )
            model.load_state_dict(pretrained_model.state_dict())
            model.to(device)
            return model_cfg["trainer"]["lr"]

        # check if dir is empty
        return module, [ckpt_id], load_state_dict

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
        bench_save_dir: str,
        load_meta_from_disk: bool = True,
    ) -> torch.utils.data.Dataset:
        """Load dataset based on configuration."""
        if "single_class_dataset" not in ds_config:
            return cls._load_hf_dataset(ds_config, load_meta_from_disk)
        elif ds_config["single_class_dataset"]:
            return cls._load_single_class_dataset(
                ds_config, bench_save_dir=bench_save_dir
            )
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
    def _load_single_class_dataset(
        cls, ds_config: dict, bench_save_dir: str
    ) -> torch.utils.data.Dataset:
        """Load a dataset from a zip file based on configuration."""
        dataset_dir = os.path.join(bench_save_dir, ds_config["save_dir"])
        transform = cls._get_transform(ds_config)
        transform = transform if transform is not None else lambda x: x
        base_dataset = load_dataset(
            ds_config["dataset_str"],
            split=ds_config.get("dataset_split", "train"),
            cache_dir=dataset_dir,
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
        bench_save_dir = ds_config.get("bench_save_dir", ".tmp")
        metadata_dir = os.path.join(bench_save_dir, "metadata")
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
