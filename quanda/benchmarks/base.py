"""Base class for all benchmarks."""

import os
import warnings
from abc import ABC
from typing import Callable, List, Optional, Any

import torch
from tqdm import tqdm

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.explainers import Explainer
from quanda.metrics import Metric
from quanda.utils.common import (
    get_load_state_dict_func,
    load_last_checkpoint,
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
        self.checkpoints_load_func: Optional[Callable[..., Any]]
        self.use_predictions: bool = False

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
        obj.train_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("train_dataset"),
            metadata_dir=config.get("metadata_dir", "./tmp"),
            dataset_dir=config.get("dataset_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )
        obj.val_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("val_dataset", None),
            metadata_dir=config.get("metadata_dir", "./tmp"),
            dataset_dir=config.get("dataset_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )
        obj.eval_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("eval_dataset"),
            metadata_dir=config.get("metadata_dir", "./tmp"),
            dataset_dir=config.get("dataset_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )

        obj.model, obj.checkpoints = BenchConfigParser.parse_model_cfg(
            model_cfg=config["model"],
            checkpoint_path=config["ckpt_dir"],
            cfg_id=config["id"],
        )
        obj.checkpoints_load_func = None  # TODO: be more flexible
        return obj

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

        # Parse trainer configuration
        trainer = BenchConfigParser.parse_trainer_cfg(config["trainer"])

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

        ckpt_dir = BenchConfigParser.get_ckpt_folder(
            config["model"], config["ckpt_dir"], config["id"]
        )
        if len(os.listdir(ckpt_dir)) > 0:
            warnings.warn(
                f"Directory {ckpt_dir} already exists and is not empty. "
                "Checkpoints will be overwritten."
            )
            # remove existing checkpoints
            for file in os.listdir(ckpt_dir):
                os.remove(os.path.join(ckpt_dir, file))

        torch.save(
            obj.model.state_dict(),
            os.path.join(ckpt_dir, f"{config['id']}.pth"),
        )

        obj.model.to(obj.device)
        obj.model.eval()
        obj.checkpoints = [os.path.join(ckpt_dir, f"{config['id']}.pth")]

        # obj.save_metadata() TODO: implement this

        return obj

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
