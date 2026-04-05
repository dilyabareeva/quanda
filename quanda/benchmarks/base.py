"""Base class for all benchmarks."""

import os
import warnings
from abc import ABC
from typing import Any, Callable, List, Optional, Union

import datasets  # type: ignore
import lightning as L
import torch
import yaml
from huggingface_hub import PyTorchModelHubMixin, create_repo, upload_folder
from tqdm import tqdm

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.resources.config_map import config_map
from quanda.explainers import Explainer
from quanda.metrics import Metric
from quanda.utils.common import DatasetSplit, class_accuracy, load_last_checkpoint
from quanda.utils.datasets.dataset_handlers import get_dataset_handler


class Benchmark(ABC):
    """Base class for all benchmarks."""

    name: str
    eval_args: List = []

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        checkpoints: List[str],
        checkpoints_load_func: Callable[..., Any],
        device: str = "cpu",
        val_dataset: Optional[
            Union[torch.utils.data.Dataset, datasets.Dataset]
        ] = None,
        use_predictions: bool = False,
    ):
        """Initialize the base `Benchmark` class.

        Parameters
        ----------
        model : torch.nn.Module
            The model to evaluate.
        train_dataset : Union[torch.utils.data.Dataset, datasets.Dataset]
            The training dataset.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset.
        checkpoints : List[str]
            List of checkpoint paths.
        checkpoints_load_func : Callable[..., Any]
            Function to load model checkpoints.
        device : str, optional
            Device to use, by default "cpu".
        val_dataset : Optional[Union[torch.utils.data.Dataset,
            datasets.Dataset]], optional
            The validation dataset, by default None.
        use_predictions : bool, optional
            Whether to use model predictions as targets, by default False.

        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.checkpoints = checkpoints
        self.checkpoints_load_func = checkpoints_load_func
        self.device = device
        self.val_dataset = val_dataset
        self.use_predictions = use_predictions

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

        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=cfg, bench_save_dir=cache_dir
        )
        BenchConfigParser.load_metadata(
            cfg,
            metadata_dir,
        )
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
    ) -> "Benchmark":
        """Initialize the benchmark from a dictionary."""
        cache_dir = config.get("bench_save_dir", "./tmp")
        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=config, bench_save_dir=cache_dir
        )
        train_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("train_dataset"),
            metadata_dir=metadata_dir,
            load_meta_from_disk=load_meta_from_disk,
        )

        # If val shares the same split file and dataset_split as
        # train, reuse the split that train just generated instead
        # of creating a new one.
        val_cfg = config.get("val_dataset", None)
        val_load_meta = load_meta_from_disk
        if val_cfg is not None and not load_meta_from_disk:
            train_cfg = config.get("train_dataset", {})
            train_indices = train_cfg.get("indices", {})
            val_indices = val_cfg.get("indices", {})
            if (
                isinstance(train_indices, dict)
                and isinstance(val_indices, dict)
                and train_indices.get("split_filename")
                == val_indices.get("split_filename")
                and train_cfg.get("dataset_split")
                == val_cfg.get("dataset_split")
            ):
                val_load_meta = True

        val_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=val_cfg,
            metadata_dir=metadata_dir,
            load_meta_from_disk=val_load_meta,
        )
        eval_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("eval_dataset"),
            metadata_dir=metadata_dir,
            load_meta_from_disk=load_meta_from_disk,
        )

        model, checkpoints, checkpoints_load_func = (
            BenchConfigParser.parse_model_cfg(
                model_cfg=config["model"],
                bench_save_dir=config["bench_save_dir"],
                repo_id=config["repo_id"],
                ckpts=config["ckpts"],
                load_model_from_disk=offline,
                device=device,
            )
        )

        extra = cls._extra_kwargs_from_config(
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            metadata_dir=metadata_dir,
            load_meta_from_disk=load_meta_from_disk,
        )

        return cls(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            device=device,
            val_dataset=val_dataset,
            use_predictions=config.get("use_predictions", True),
            **extra,
        )

    @classmethod
    def _extra_kwargs_from_config(
        cls,
        config: dict,
        train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        metadata_dir: str,
        load_meta_from_disk: bool,
    ) -> dict:
        """Extract subclass-specific kwargs from config.

        Override in subclasses to provide additional constructor arguments.

        Parameters
        ----------
        config : dict
            The benchmark configuration dictionary.
        train_dataset : Union[torch.utils.data.Dataset, datasets.Dataset]
            The parsed training dataset.
        eval_dataset : torch.utils.data.Dataset
            The parsed evaluation dataset.
        metadata_dir : str
            Path to the metadata directory.
        load_meta_from_disk : bool
            Whether metadata was loaded from disk.

        Returns
        -------
        dict
            Additional keyword arguments for the constructor.

        """
        return {}

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

        accelerator = "gpu" if "cuda" in obj.device else obj.device
        trainer.fit(
            model=obj.model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            accelerator=accelerator,
        )

        ckpt_dir = os.path.join(
            config.get("bench_save_dir", "./tmp"),
            "ckpt",
            config["ckpts"][-1],
        )

        os.makedirs(ckpt_dir, exist_ok=True)
        if len(os.listdir(ckpt_dir)) > 0:
            warnings.warn(
                f"Directory {ckpt_dir} already exists and is not empty. "
                "Checkpoints will be overwritten."
            )

        obj.model.to(obj.device)
        obj.model.eval()

        assert isinstance(obj.model, PyTorchModelHubMixin), (
            "Model must inherit from PyTorchModelHubMixin."
        )
        obj.model.save_pretrained(ckpt_dir, safe_serialization=True)
        obj.checkpoints = [ckpt_dir]

        obj._compute_and_save_indices(config, batch_size)

        return obj

    @classmethod
    def train_and_push_to_hub(
        cls,
        config: dict,
        logger: Optional[L.pytorch.loggers.logger.Logger] = None,
        device: str = "cpu",
        batch_size: int = 8,
    ):  # pragma: no cover
        """Train a model using the provided config and push to HF hub."""
        obj = cls.train(
            config,
            logger=logger,
            device=device,
            batch_size=batch_size,
        )
        assert isinstance(obj.model, PyTorchModelHubMixin), (
            "Model must inherit from PyTorchModelHubMixin."
        )

        # TODO: add support for multiple checkpoints
        obj.model.push_to_hub(f"quanda-bench-test/{config['ckpts'][-1]}")

        # TODO: push to hub for LDS models

        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=config, bench_save_dir=config.get("bench_save_dir", "./tmp")
        )
        create_repo(
            repo_id=f"quanda-bench-test/{config['id']}_metadata",
            repo_type="dataset",
            exist_ok=True,
        )
        upload_folder(
            folder_path=metadata_dir,
            repo_id=f"quanda-bench-test/{config['id']}_metadata",
            repo_type="dataset",
        )

        return obj

    def _compute_and_save_indices(self, config: dict, batch_size: int = 8):
        """Determine the indices of eval dataset, if needed. By default, all samples are kept.

        Parameters
        ----------
        config : dict
            Benchmark configuration dictionary (needed for save path).
        batch_size : int, optional
            Batch size for the inference pass, by default 8.

        """
        return
        
        
    def _compute_and_save_filter_by_class_prediction(
        self, 
        config: dict, 
        batch_size: int = 8,
        filter_by_class: bool = False,
        filter_cls: Optional[int] = None,
        shortcut_cls: Optional[int] = None,
        filter_by_non_shortcut: bool = False,
        filter_by_prediction: bool = False,
    ):
        """Run inference on eval_dataset and save the filter correctly predicted indices.

        Iterates over ``self.eval_dataset``, calls ``_compute_filter_mask``
        on every batch, collects the selected indices, stores them in
        ``self.filter_indices``, and persists them via
        ``save_filtered_indices``.

        Parameters
        ----------
        config : dict
            Benchmark configuration dictionary (needed for save path).
        batch_size : int, optional
            Batch size for the inference pass, by default 8.

        """
        ds_handler = get_dataset_handler(dataset=self.eval_dataset)
        expl_dl = ds_handler.create_dataloader(
            dataset=self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        if filter_by_class and filter_cls is None:
            raise ValueError(
                "filter_cls must be provided if filter_by_class is True."
            )
            
        if filter_by_non_shortcut and shortcut_cls is None:
            raise ValueError(
                "shortcut_cls must be provided if filter_by_non_shortcut is True."
            )

        select_indices: list = []
        for batch in expl_dl:
            inputs, labels = ds_handler.process_batch(
                batch=batch, device=self.device
            )
            model_inputs = ds_handler.get_model_inputs(inputs=inputs)
            outputs = (
                self.model(**model_inputs)
                if isinstance(model_inputs, dict)
                else self.model(model_inputs)
            )
            pred_cls = ds_handler.get_predictions(outputs=outputs)
            select_idx = torch.tensor([True] * len(pred_cls)).to(inputs.device)
            if filter_by_class:
                select_idx *= labels != filter_cls
            if filter_by_non_shortcut:
                select_idx *= pred_cls == shortcut_cls
            if filter_by_prediction:
                select_idx *= pred_cls == labels
            select_indices.extend(select_idx)

        filter_indices = torch.nonzero(
            torch.tensor(select_indices), as_tuple=False
        )
        self.save_filtered_indices(config, filter_indices)

    def save_filtered_indices(self, config: dict, filter_indices: torch.Tensor):
        """Persist ``filter_indices`` to the metadata directory.

        Reads the filter-indices filename from ``config['eval_dataset']
        ['filter_indices']`` or ``config['eval_filter_indices']`` and
        saves a :class:`~quanda.utils.common.DatasetSplit` YAML file.

        Parameters
        ----------
        config : dict
            Benchmark configuration dictionary.
        filter_indices : torch.Tensor
            Tensor containing the indices of the filtered samples.

        """
        cache_dir = config.get("bench_save_dir", "./tmp")
        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=config, bench_save_dir=cache_dir
        )
        
        eval_ds_cfg = config.get("eval_dataset", {})
        if "filter_indices" not in eval_ds_cfg:
            raise ValueError(
                "Filter indices filename must be specified in config under "
                "'eval_dataset.filter_indices' or 'eval_filter_indices'."
            )
        filter_cfg = eval_ds_cfg.get("filter_indices")
        split = DatasetSplit({filter_cfg["split_name"]: filter_indices})
        split.save(metadata_dir, filter_cfg["split_filename"])
        
        if self.eval_dataset.transform_indices is not None:
            # only keep the filtered indices in the transformed dataset
            filter_set = set(filter_indices.flatten().tolist())
            self.eval_dataset.transform_indices = [
                idx for idx in self.eval_dataset.transform_indices
                if idx in filter_set
            ]
            self.eval_dataset.metadata.transform_indices = (
                self.eval_dataset.transform_indices
            )
            # save new transform indices into metadata dir
            wrapper_cfg = config.get("eval_dataset", {}).get(
                "wrapper", {}
            )
            meta_filename = wrapper_cfg.get("metadata", {}).get(
                "metadata_filename"
            )
            if meta_filename is not None:
                self.eval_dataset.metadata.save(
                    metadata_dir, meta_filename
                )

    def load_last_checkpoint(self):
        """Load the last checkpoint into the model."""

        load_last_checkpoint(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
        )
        
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

        self.load_last_checkpoint()
        
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
        # TODO: Should we always require a checkpoint?
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
        """Evaluate dataset using explainer and metric.

        Parameters
        ----------
        eval_dataset : torch.utils.data.Dataset
            Dataset to evaluate
        explainer : Explainer
            Explainer to use for generating explanations
        metric : Metric
            Metric to compute
        batch_size : int
            Batch size for evaluation

        Returns
        -------
        Any
            Computed metric result

        """
        ds_handler = get_dataset_handler(dataset=eval_dataset)
        expl_dl = ds_handler.create_dataloader(
            dataset=eval_dataset,
            batch_size=batch_size,
        )

        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, batch in enumerate(pbar):
            pbar.set_description(
                f"Metric evaluation, batch {i + 1}/{n_batches}"
            )

            inputs, labels = ds_handler.process_batch(
                batch=batch,
                device=self.device,
            )

            if self.use_predictions:
                with torch.no_grad():
                    model_inputs = ds_handler.get_model_inputs(inputs=inputs)
                    outputs = (
                        self.model(**model_inputs)
                        if isinstance(model_inputs, dict)
                        else self.model(model_inputs)
                    )
                    targets = ds_handler.get_predictions(outputs=outputs)
            else:
                targets = labels

            explanations = explainer.explain(
                test_data=inputs,
                targets=targets,
            )

            data_unit = {
                "test_data": inputs,
                "test_targets": targets,
                "test_labels": labels,
                "explanations": explanations,
            }

            if hasattr(self, "class_to_group"):
                data_unit["test_targets"] = torch.tensor(
                    [self.class_to_group[i.item()] for i in labels],
                    device=labels.device,
                )
                if not self.use_predictions:
                    data_unit["targets"] = data_unit["grouped_labels"]

            eval_unit = {k: data_unit[k] for k in self.eval_args}
            metric.update(**eval_unit)

        return metric.compute()
