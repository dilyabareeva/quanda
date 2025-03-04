"""Base class for all benchmarks."""

import os
import warnings
from abc import ABC
from typing import Callable, List, Optional, Any

import lightning as L
import torch
import yaml
from tqdm import tqdm
from huggingface_hub import upload_folder, create_repo

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.resources.config_map import config_map
from quanda.explainers import Explainer
from quanda.metrics import Metric
from quanda.utils.common import (
    load_last_checkpoint,
    class_accuracy,
)


class Benchmark(ABC):
    """Base class for all benchmarks."""

    name: str
    eval_args: List = []

    def __init__(self, *args, **kwargs):
        """Initialize the base `Benchmark` class."""
        self.model: torch.nn.Module
        self.device: str = "cpu"
        self.train_dataset: torch.utils.data.Dataset
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.eval_dataset: torch.utils.data.Dataset
        self.checkpoints: List[str] = []
        self.checkpoints_load_func: Callable[..., Any]
        self.use_predictions: bool = False

    @classmethod
    def load_pretrained(
        cls,
        bench_id: str,
        cache_dir: str,
        device: str = "cpu",
        offline: bool = False,
    ):
        """Load a precomputed benchmark.

        Load precomputed benchmark components from a file and creates an
        instance from the state dictionary.

        Parameters
        ----------
        bench_id : str
            ID of the benchmark to be loaded.
        cache_dir : str
            Directory to store the downloaded benchmark components.
        device : str, optional
            Device to load the model on, by default "cpu".
        offline : bool, optional
            Whether to load the benchmark in offline mode, by default False.

        Returns
        -------
        Benchmark
            The benchmark instance.

        """
        bench_yaml = config_map[bench_id]

        # Load the benchmark configuration
        with open(bench_yaml, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["bench_save_dir"] = cache_dir
        return cls.from_config(
            cfg,
            load_meta_from_disk=True,
            offline=offline,
            device=device,
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        load_meta_from_disk: bool = True,
        offline: bool = False,
        device: str = "cpu",
    ):
        """Initialize the benchmark from a dictionary."""
        obj = cls()
        obj.device = device
        metadata_dir = BenchConfigParser.load_metadata(
            cfg=config,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )
        obj.train_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("train_dataset"),
            metadata_dir=metadata_dir,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )
        obj.val_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("val_dataset", None),
            metadata_dir=metadata_dir,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )
        obj.eval_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("eval_dataset"),
            metadata_dir=metadata_dir,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )

        obj.model, obj.checkpoints, obj.checkpoints_load_func = (
            BenchConfigParser.parse_model_cfg(
                model_cfg=config["model"],
                bench_save_dir=config["bench_save_dir"],
                repo_id=config["repo_id"],
                cfg_id=config["id"],
                offline=offline,
                device=device,
            )
        )

        return obj

    @classmethod
    def train(
        cls,
        config: dict,
        logger: Optional[L.pytorch.loggers.logger.Logger] = None,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> "Benchmark":
        """Train a model using the provided configuration.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.
        logger : Optional[Callable], optional
            Logger to be used for logging, by default None.
        device : str, optional
            Device to use for training, by default "cpu"
        batch_size : int, optional
            Batch size for training, by default 8

        Returns
        -------
        None

        """
        obj = cls.from_config(config, load_meta_from_disk=False, device=device)

        # Parse trainer configuration
        trainer = BenchConfigParser.parse_trainer_cfg(
            config["model"]["trainer"]
        )
        if logger is not None:
            trainer.logger = logger

        train_dl = torch.utils.data.DataLoader(
            obj.train_dataset,
            batch_size=batch_size,
            shuffle=False,  # TODO: true
        )
        if obj.val_dataset is not None:
            val_dl = torch.utils.data.DataLoader(
                obj.val_dataset,
                batch_size=batch_size,
                shuffle=False,  # TODO: true
            )
        else:
            val_dl = None

        obj.model.train()
        obj.model.to(obj.device)

        trainer.fit(
            model=obj.model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
        )

        ckpt_dir = os.path.join(config["bench_save_dir"], "ckpt")
        ckpt_dir = BenchConfigParser.get_ckpt_folder(
            config["model"], ckpt_dir, config["id"]
        )
        if len(os.listdir(ckpt_dir)) > 0:
            warnings.warn(
                f"Directory {ckpt_dir} already exists and is not empty. "
                "Checkpoints will be overwritten."
            )

        obj.model.to(obj.device)
        obj.model.eval()

        hf_ckpt_dir = ckpt_dir
        obj.model.save_pretrained(hf_ckpt_dir, safe_serialization=True)
        obj.checkpoints = [hf_ckpt_dir]

        return obj

    @classmethod
    def train_and_push_to_hub(
        cls,
        config: dict,
        logger: Optional[L.pytorch.loggers.logger.Logger] = None,
        device: str = "cpu",
        batch_size: int = 8,
    ):
        """Train a model using the provided config and push to HF hub."""
        obj = cls.train(
            config,
            logger=logger,
            device=device,
            batch_size=batch_size,
        )
        obj.model.push_to_hub(f"quanda-bench-test/{config['id']}")

        metadata_dir = BenchConfigParser.load_metadata(
            cfg=config,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            load_meta_from_disk=False,
        )
        create_repo(
            repo_id=config["metadata_str"], repo_type="dataset", exist_ok=True
        )
        upload_folder(
            folder_path=metadata_dir,
            repo_id=config["metadata_str"],
            repo_type="dataset",
        )

        return obj

    def sanity_check(self, batch_size: int = 32) -> dict:
        """Compute training and validation accuracy of the model.

        Parameters
        ----------
        batch_size : int, optional
            Batch size to be used for the evaluation, defaults to 32.

        Returns
        -------
        dict
            Computed accuracy results.

        """
        results = {}

        train_dl = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        if self.val_dataset is not None:
            val_dl = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            val_dl = None

        self.model.eval()
        self.model.to(self.device)

        # By default we only run accuracy
        results["train_acc"] = class_accuracy(
            self.model, train_dl, self.device
        )

        if val_dl is not None:
            results["val_acc"] = class_accuracy(
                self.model, val_dl, self.device
            )

        return results

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

    def save_metadata(self):
        """Save metadata to disk."""
        raise NotImplementedError

    def _prepare_explainer(
        self,
        dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
    ):
        if len(self.checkpoints) == 0:
            raise ValueError(
                "No model checkpoints found. Use `train` method "
                "to train the model."
            )
        load_last_checkpoint(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
        )
        self.model.eval()
        self.model.to(self.device)

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            **expl_kwargs,
        )
        return explainer

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

            if hasattr(self, "class_to_group"):
                data_unit["grouped_labels"] = torch.tensor(
                    [self.class_to_group[i.item()] for i in labels],
                    device=labels.device,
                )
                if not self.use_predictions:
                    data_unit["targets"] = data_unit["grouped_labels"]

            eval_unit = {k: data_unit[k] for k in self.eval_args}
            metric.update(**eval_unit)

        return metric.compute()
