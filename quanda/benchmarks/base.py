"""Base class for all benchmarks."""

import hashlib
import json
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

import datasets  # type: ignore
import lightning as L
import torch
import yaml
from huggingface_hub import (
    PyTorchModelHubMixin,
    create_branch,
    create_repo,
    snapshot_download,
    upload_folder,
)
from tqdm import tqdm

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.resources.config_map import config_map
from quanda.explainers import Explainer
from quanda.metrics import Metric
from quanda.utils.cache import BatchedCachedExplanations, ExplanationsCache
from quanda.utils.common import (
    DatasetSplit,
    _stable_repr,
    _subsample_dataset,
    chunked_logits,
    class_accuracy,
    load_last_checkpoint,
)
from quanda.utils.datasets.dataset_handlers import get_dataset_handler
from quanda.utils.datasets.transformed.base import TransformedDataset
from quanda.utils.training.trainer import _EpochSnapshotCallback


def _hash_expl_kwargs(expl_kwargs: Optional[dict]) -> str:
    """Stable short hash of sorted expl_kwargs for explanation repo IDs."""
    payload = json.dumps(
        expl_kwargs or {}, sort_keys=True, default=_stable_repr
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def _resolve_ckpts(config: dict) -> List[str]:
    """Expand ``config['ckpt']`` + ``num_checkpoints`` into a ckpt list.

    Returns a single-entry list when ``num_checkpoints <= 1`` and a
    list of ``<repo>@epoch_<i>`` revision-suffixed entries otherwise.
    """
    repo_id = config["ckpt"]
    n = int(config.get("num_checkpoints", 1))
    if n <= 1:
        return [repo_id]
    return [f"{repo_id}@epoch_{i + 1}" for i in range(n)]


def default_explanations_id(
    config: dict,
    explainer_cls: type,
    expl_kwargs: Optional[dict],
    max_eval_n: Optional[int] = 1000,
    eval_seed: int = 42,
) -> str:
    """Build the default HF repo_id for cached explanations.

    ``max_eval_n`` and ``eval_seed`` are encoded in the id so that cached
    explanations stay coupled to the exact eval-dataset subsample they
    were computed on. For benchmarks driven by training-data
    self-influence (e.g. MislabelingDetection), these same parameters
    describe the train-dataset subsample instead.

    If ``config['explanations_group']`` is set, it replaces ``config['id']``
    as the identity segment so multiple benchmarks that share the same
    model + train/eval datasets (e.g. qnli ClassDetection and LDS) can
    reuse a single cached explanations artifact. Only opt in when the
    grouped benchmarks truly share those inputs — a mismatch will be
    silent.
    """
    repo = config.get("repo_id", "quanda-bench-test")
    group = config.get("explanations_group", config["id"])
    return (
        f"{repo}/{group}__{explainer_cls.__name__}"
        f"__{_hash_expl_kwargs(expl_kwargs)}"
        f"__n{max_eval_n}_s{eval_seed}_explanations"
    )


class Benchmark(ABC):
    """Base class for all benchmarks."""

    name: str
    eval_args: List = []
    default_use_predictions: bool = False

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

        self._pid_suffix: str = ""
        self._explanations_id: Optional[str] = None
        self._explanations_dir: Optional[str] = None

    @classmethod
    def load_pretrained(
        cls,
        bench_id: str,
        cache_dir: str,
        device: str = "cpu",
        offline: bool = False,
        load_fresh: bool = False,
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
            If True, no HTTP request is issued to the Hub; all assets
            (metadata, model) must already be present under
            ``cache_dir``. By default False.
        load_fresh : bool, optional
            If True, re-download metadata and model from the Hub,
            overwriting the local cache. Incompatible with
            ``offline=True``. By default False.

        Returns
        -------
        Benchmark
            The benchmark instance.

        """
        if offline and load_fresh:
            raise ValueError(
                "offline=True and load_fresh=True are incompatible: "
                "cannot refresh the cache without network access."
            )

        bench_yaml = (
            bench_id if os.path.isfile(bench_id) else config_map[bench_id]
        )

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
            offline=offline,
            load_fresh=load_fresh,
        )
        obj = cls.from_config(
            cfg,
            load_meta_from_disk=True,
            offline=offline,
            load_fresh=load_fresh,
            device=device,
        )

        return obj

    @classmethod
    def from_config(
        cls,
        config: dict,
        load_meta_from_disk: bool = True,
        offline: bool = False,
        device: str = "cpu",
        metadata_suffix: str = "",
        load_fresh: bool = False,
    ) -> "Benchmark":
        """Initialize the benchmark from a dictionary."""
        if offline and load_fresh:
            raise ValueError(
                "offline=True and load_fresh=True are incompatible."
            )
        cache_dir = config.get("bench_save_dir", "./tmp")
        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=config,
            bench_save_dir=cache_dir,
            suffix=metadata_suffix,
        )
        splits_cfg = config.get("splits", {})
        train_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("train_dataset"),
            metadata_dir=metadata_dir,
            load_meta_from_disk=load_meta_from_disk,
            splits_cfg=splits_cfg,
        )

        val_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("val_dataset"),
            metadata_dir=metadata_dir,
            load_meta_from_disk=load_meta_from_disk,
            splits_cfg=splits_cfg,
        )
        eval_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("eval_dataset"),
            metadata_dir=metadata_dir,
            load_meta_from_disk=load_meta_from_disk,
            splits_cfg=splits_cfg,
        )

        model, checkpoints, checkpoints_load_func = (
            BenchConfigParser.parse_model_cfg(
                model_cfg=config["model"],
                bench_save_dir=config["bench_save_dir"],
                ckpts=_resolve_ckpts(config),
                offline=offline,
                load_fresh=load_fresh,
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
            use_predictions=config.get(
                "use_predictions", cls.default_use_predictions
            ),
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
        batch_size: int = 64,
        load_meta_from_disk: bool = False,
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
        load_meta_from_disk : bool, optional
            If True, reuse existing metadata (splits, class mappings,
            etc.) from the cache instead of regenerating. By default
            False — training regenerates metadata so that a fresh
            training run is reproducible from the config alone.

        Returns
        -------
        None

        """
        pid_suffix = f"_pid{os.getpid()}"
        obj = cls.from_config(
            config,
            load_meta_from_disk=load_meta_from_disk,
            device=device,
            metadata_suffix=pid_suffix,
        )
        obj._pid_suffix = pid_suffix

        pretrained_base = BenchConfigParser.load_pretrained_base(
            model_cfg=config["model"], device=device
        )
        if pretrained_base is not None:
            obj.model = pretrained_base

        # Parse trainer configuration
        trainer = BenchConfigParser.parse_trainer_cfg(
            config["model"]["trainer"]
        )
        if logger is not None:
            trainer.logger = logger

        ds_handler = get_dataset_handler(dataset=obj.train_dataset)
        train_dl = ds_handler.create_dataloader(
            dataset=obj.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=trainer.num_workers,
        )
        if obj.val_dataset is not None:
            val_ds_handler = get_dataset_handler(dataset=obj.val_dataset)
            val_dl = val_ds_handler.create_dataloader(
                dataset=obj.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=trainer.num_workers,
            )
        else:
            val_dl = None

        obj.model.train()
        obj.model.to(obj.device)

        if "cuda" in obj.device:
            accelerator = "gpu"
            devices = (
                int(obj.device.split(":")[-1]) if ":" in obj.device else 0
            )
        else:
            accelerator = obj.device
            devices = 1

        ckpt_dir = os.path.join(
            config.get("bench_save_dir", "./tmp"),
            "ckpt",
            f"{config['ckpt'].split('/')[-1]}{pid_suffix}",
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        if len(os.listdir(ckpt_dir)) > 0:
            warnings.warn(
                f"Directory {ckpt_dir} already exists and is not empty. "
                "Checkpoints will be overwritten."
            )

        num_checkpoints = int(config.get("num_checkpoints", 1))
        snapshot_dirs: List[str] = []
        callbacks: Optional[List[L.Callback]] = None
        if num_checkpoints > 1:
            max_epochs = config["model"]["trainer"]["max_epochs"]
            snapshot_epochs = sorted(
                {
                    min(
                        max_epochs - 1,
                        int((i + 1) * max_epochs / num_checkpoints) - 1,
                    )
                    for i in range(num_checkpoints)
                }
            )
            snapshot_dirs = [
                os.path.join(ckpt_dir, f"epoch_{i + 1}")
                for i in range(len(snapshot_epochs))
            ]
            callbacks = [
                _EpochSnapshotCallback(snapshot_epochs, snapshot_dirs)
            ]

        trainer.fit(
            model=obj.model,
            train_dataloaders=train_dl,
            val_dataloaders=val_dl,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
        )

        obj.model.to(obj.device)
        obj.model.eval()

        if not isinstance(obj.model, PyTorchModelHubMixin):
            raise TypeError("Model must inherit from PyTorchModelHubMixin.")
        if snapshot_dirs:
            obj.checkpoints = snapshot_dirs
        else:
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
        batch_size: int = 64,
        load_meta_from_disk: bool = False,
    ):  # pragma: no cover
        """Train a model using the provided config and push to HF hub."""
        skip_main_train = bool(config.get("skip_main_train", False))
        if skip_main_train:
            obj = cls.from_config(
                config,
                load_meta_from_disk=load_meta_from_disk,
                device=device,
            )
            obj._compute_and_save_indices(config, batch_size)
        else:
            obj = cls.train(
                config,
                logger=logger,
                device=device,
                batch_size=batch_size,
                load_meta_from_disk=load_meta_from_disk,
            )
            if not isinstance(obj.model, PyTorchModelHubMixin):
                raise TypeError(
                    "Model must inherit from PyTorchModelHubMixin."
                )

            repo_id = config["ckpt"]
            num_checkpoints = int(config.get("num_checkpoints", 1))
            if num_checkpoints <= 1:
                obj.model.push_to_hub(repo_id)
            else:
                create_repo(repo_id=repo_id, exist_ok=True)
                for i, snapshot_dir in enumerate(obj.checkpoints, start=1):
                    revision = f"epoch_{i}"
                    create_branch(
                        repo_id=repo_id, branch=revision, exist_ok=True
                    )
                    upload_folder(
                        folder_path=snapshot_dir,
                        repo_id=repo_id,
                        revision=revision,
                    )
                upload_folder(
                    folder_path=obj.checkpoints[-1],
                    repo_id=repo_id,
                )

        pid_suffix = getattr(obj, "_pid_suffix", "")
        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=config,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            suffix=pid_suffix,
        )
        meta_id = config.get(
            "meta_id", f"{config['repo_id']}/{config['id']}_metadata"
        )
        create_repo(
            repo_id=meta_id,
            repo_type="dataset",
            exist_ok=True,
        )
        upload_folder(
            folder_path=metadata_dir,
            repo_id=meta_id,
            repo_type="dataset",
        )

        return obj

    def _compute_and_save_indices(self, config: dict, batch_size: int = 8):
        """Determine the indices of eval dataset, if needed.

        By default, all samples are kept.

        Parameters
        ----------
        config : dict
            Benchmark configuration dictionary (needed for save path).
        batch_size : int, optional
            Batch size for the inference pass, by default 8.

        """
        return

    def _compute_and_save_filter_by_labels_and_prediction(
        self,
        config: dict,
        batch_size: int = 8,
        filter_by_shortcut_pred: bool = False,
        shortcut_cls: Optional[int] = None,
        filter_by_non_shortcut: bool = False,
        filter_by_prediction: bool = False,
    ):
        """Run inference on eval_dataset and save filtered indices.

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
        filter_by_shortcut_pred : bool, optional
            Whether to filter by shortcut prediction,
            by default False.
        shortcut_cls : Optional[int], optional
            The shortcut class index, by default None.
        filter_by_non_shortcut : bool, optional
            Whether to filter non-shortcut samples,
            by default False.
        filter_by_prediction : bool, optional
            Whether to filter by correct prediction,
            by default False.

        """
        ds_handler = get_dataset_handler(dataset=self.eval_dataset)
        expl_dl = ds_handler.create_dataloader(
            dataset=self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        if not (
            filter_by_shortcut_pred
            or filter_by_non_shortcut
            or filter_by_prediction
        ):
            return

        if filter_by_shortcut_pred and shortcut_cls is None:
            raise ValueError(
                "shortcut_cls must be provided if "
                "filter_by_shortcut_pred is True."
            )

        if filter_by_non_shortcut and shortcut_cls is None:
            raise ValueError(
                "shortcut_cls must be provided if "
                "filter_by_non_shortcut is True."
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
            select_idx = torch.tensor([True] * len(pred_cls)).to(self.device)
            if filter_by_shortcut_pred:
                select_idx *= pred_cls == shortcut_cls
            if filter_by_non_shortcut:
                select_idx *= labels != shortcut_cls
            if filter_by_prediction:
                select_idx *= pred_cls == labels
            select_indices.extend(select_idx)

        filter_indices = [
            i for i in range(len(select_indices)) if select_indices[i] != 0
        ]
        self.save_filtered_indices(config, filter_indices)
        if isinstance(self.eval_dataset, TransformedDataset):
            self.eval_dataset.apply_filter(filter_indices)
        else:
            self.eval_dataset = torch.utils.data.Subset(
                self.eval_dataset, filter_indices
            )

    def save_filtered_indices(self, config: dict, filter_indices: list):
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
        pid_suffix = getattr(self, "_pid_suffix", "")
        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=config,
            bench_save_dir=cache_dir,
            suffix=pid_suffix,
        )

        eval_ds_cfg = config.get("eval_dataset", {})
        if "filter_indices" not in eval_ds_cfg:
            raise ValueError(
                "Filter indices filename must be specified in config under "
                "'eval_dataset.filter_indices' or 'eval_filter_indices'."
            )
        filter_cfg = eval_ds_cfg.get("filter_indices")
        split = DatasetSplit(
            {filter_cfg["split_name"]: torch.tensor(filter_indices)}
        )
        split.save(metadata_dir, filter_cfg["split_filename"])

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

        train_handler = get_dataset_handler(dataset=self.train_dataset)
        train_dl = train_handler.create_dataloader(
            dataset=self.train_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
        if self.val_dataset is not None:
            val_handler = get_dataset_handler(dataset=self.val_dataset)
            val_dl = val_handler.create_dataloader(
                dataset=self.val_dataset,
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

    def overall_objective(self, sanity_check_results: dict) -> float:
        """Compute overall objective score.

        Based on sanity check results, for selecting optional
        hyperparameters of the benchmark.
        By default, this method can be used to compute an overall
        score from the sanity check results.

        Parameters
        ----------
        sanity_check_results : dict
            Dictionary containing the results from the sanity check.

        Returns
        -------
        float
            Overall objective score computed from the sanity check results.

        """
        return sum(sanity_check_results.values()) / len(sanity_check_results)

    @abstractmethod
    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        max_eval_n: Optional[int] = 1000,
        eval_seed: int = 42,
        cache_dir: Optional[str] = None,
        use_cached_expl: bool = False,
        use_hf_expl: bool = False,
        inference_batch_size: Optional[int] = None,
    ):
        """Run the evaluation using the benchmark.

        Parameters
        ----------
        explainer_cls : type
            Explainer subclass to instantiate for this evaluation.
        expl_kwargs : Optional[dict]
            Extra kwargs passed through to ``explainer_cls``.
        batch_size : int
            Batch size used when iterating the eval dataset.
        max_eval_n : Optional[int]
            Cap on the number of eval samples; ``None`` means all.
        eval_seed : int
            Seed used when sampling the eval subset.
        cache_dir : Optional[str]
            Directory where explanations are cached on disk. Required when
            ``use_cached_expl`` or ``use_hf_expl`` is ``True``.
        use_cached_expl : bool
            Load precomputed explanations from ``cache_dir`` instead of
            recomputing them.
        use_hf_expl : bool
            Download precomputed explanations from the HF Hub into
            ``cache_dir`` before loading.
        inference_batch_size : Optional[int]
            If set, every forward through ``self.model`` run during
            evaluation (prediction of ``targets`` and any model calls in
            ``metric.update``) is split into sub-batches of this size.
            ``None`` keeps the full ``batch_size``-wide forward.

        Returns
        -------
        dict
            Metric scores produced by this benchmark's metric(s).

        """
        raise NotImplementedError

    def _resolve_precomputed_explanations(
        self,
        cache_dir: Optional[str],
        use_cached_expl: bool = False,
        use_hf_expl: bool = False,
    ) -> Optional[BatchedCachedExplanations]:
        """Return a cached-explanations handle if available.

        Uses the same ``cache_dir`` that was passed to :meth:`explain`. If
        ``use_cached_expl`` and ``cache_dir`` exists locally, load from it.
        Else if ``use_hf_expl``, download the HF dataset repo into
        ``cache_dir`` (repo_id derived by replacing the last ``"__"`` in
        ``basename(cache_dir)`` with ``"/"``), then load.
        """
        if not (use_cached_expl or use_hf_expl):
            return None
        if cache_dir is None:
            raise ValueError(
                "cache_dir must be provided when use_cached_expl or "
                "use_hf_expl is True."
            )
        if use_cached_expl and os.path.exists(cache_dir):
            return ExplanationsCache.load(path=cache_dir, device=self.device)
        if use_hf_expl:
            base = os.path.basename(cache_dir.rstrip("/"))
            explanations_id = (
                base[::-1].replace("__", "/", 1)[::-1]
                if "__" in base
                else base
            )
            os.makedirs(cache_dir, exist_ok=True)
            snapshot_download(
                repo_id=explanations_id,
                local_dir=cache_dir,
                repo_type="dataset",
            )
            return ExplanationsCache.load(path=cache_dir, device=self.device)
        return None

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

    @staticmethod
    def _download_explanations(
        explanations_id: str,
        cache_dir: str = ".tmp",
    ) -> str:
        """Download cached explanations and return the local directory."""
        local_dir = os.path.join(
            cache_dir, "explanations", explanations_id.replace("/", "__")
        )
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=explanations_id,
            local_dir=local_dir,
            repo_type="dataset",
        )
        return local_dir

    @classmethod
    def explain(
        cls,
        config: dict,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        explanations_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        max_eval_n: Optional[int] = 1000,
        eval_seed: int = 42,
        inference_batch_size: Optional[int] = None,
    ) -> "Benchmark":
        """Compute and persist explanations for ``eval_dataset`` to disk.

        Mirrors :meth:`train` but produces per-batch explanation tensors
        plus an ``explanations_config.yaml`` describing how the cache
        was generated. Returns the benchmark instance with
        ``self._explanations_dir`` and ``self._explanations_id`` set.
        """
        obj = cls.from_config(config, device=device)
        if explanations_id is None:
            explanations_id = default_explanations_id(
                config,
                explainer_cls,
                expl_kwargs,
                max_eval_n=max_eval_n,
                eval_seed=eval_seed,
            )

        save_dir = cache_dir or os.path.join(
            config.get("bench_save_dir", "./tmp"),
            "explanations",
            explanations_id.replace("/", "__"),
        )
        os.makedirs(save_dir, exist_ok=True)

        explainer = obj._prepare_explainer(
            dataset=obj.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        n_batches = 0
        for i, _, _, _, explanations, n_batches in obj._iter_explanations(
            explainer=explainer,
            eval_dataset=obj.eval_dataset,
            batch_size=batch_size,
            max_eval_n=max_eval_n,
            eval_seed=eval_seed,
            precomputed_explanations=None,
            inference_batch_size=inference_batch_size,
        ):
            ExplanationsCache.save(save_dir, explanations, num_id=i)

        # Repr non-serializable kwargs (e.g. callables) so YAML can dump.
        safe_kwargs = {
            k: (
                v
                if isinstance(v, (str, int, float, bool, type(None)))
                else _stable_repr(v)
            )
            for k, v in (expl_kwargs or {}).items()
        }
        meta = {
            "explanations_id": explanations_id,
            "bench_id": config.get("id"),
            "bench": config.get("bench"),
            "explainer_cls": explainer_cls.__name__,
            "expl_kwargs": safe_kwargs,
            "expl_kwargs_hash": _hash_expl_kwargs(expl_kwargs),
            "batch_size": batch_size,
            "use_predictions": obj.use_predictions,
            "n_batches": n_batches,
        }
        with open(
            os.path.join(save_dir, "explanations_config.yaml"), "w"
        ) as f:
            yaml.safe_dump(meta, f)

        obj._explanations_dir = save_dir
        obj._explanations_id = explanations_id
        return obj

    @classmethod
    def explain_and_push_to_hub(
        cls,
        config: dict,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        explanations_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        max_eval_n: Optional[int] = 1000,
        eval_seed: int = 42,
    ):  # pragma: no cover
        """Compute explanations then upload them as a HF dataset repo."""
        obj = cls.explain(
            config=config,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            batch_size=batch_size,
            explanations_id=explanations_id,
            cache_dir=cache_dir,
            device=device,
            max_eval_n=max_eval_n,
            eval_seed=eval_seed,
        )
        if obj._explanations_id is None or obj._explanations_dir is None:
            raise RuntimeError(
                "explain() must populate _explanations_id and "
                "_explanations_dir before pushing to the Hub."
            )
        create_repo(
            repo_id=obj._explanations_id,
            repo_type="dataset",
            exist_ok=True,
        )
        upload_folder(
            folder_path=obj._explanations_dir,
            repo_id=obj._explanations_id,
            repo_type="dataset",
        )
        return obj

    def _iter_explanations(
        self,
        explainer: Optional[Explainer],
        eval_dataset: torch.utils.data.Dataset,
        batch_size: int,
        max_eval_n: Optional[int],
        eval_seed: int,
        precomputed_explanations: Optional[BatchedCachedExplanations] = None,
        inference_batch_size: Optional[int] = None,
    ):
        """Yield ``(i, inputs, labels, targets, explanations, n_batches)``.

        If ``precomputed_explanations`` is provided, batch ``i`` is read from
        the cache; otherwise ``explainer.explain`` is called.
        """
        eval_dataset = _subsample_dataset(
            eval_dataset, max_n=max_eval_n, seed=eval_seed
        )
        ds_handler = get_dataset_handler(dataset=eval_dataset)
        expl_dl = ds_handler.create_dataloader(
            dataset=eval_dataset, batch_size=batch_size
        )
        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)
        for i, batch in enumerate(pbar):
            pbar.set_description(
                f"Computing eval explanations, batch {i + 1}/{n_batches}"
            )
            inputs, labels = ds_handler.process_batch(
                batch=batch, device=self.device
            )
            if self.use_predictions:
                with torch.no_grad():
                    model_inputs = ds_handler.get_model_inputs(inputs=inputs)
                    logits = chunked_logits(
                        self.model, model_inputs, inference_batch_size
                    )
                    targets = ds_handler.get_predictions(outputs=logits)
            else:
                targets = labels

            if precomputed_explanations is not None:
                explanations = precomputed_explanations[i].to(self.device)
            else:
                if explainer is None:
                    raise RuntimeError(
                        "explainer must be provided when "
                        "precomputed_explanations is None."
                    )
                explanations = explainer.explain(
                    test_data=inputs, targets=targets
                )
            yield i, inputs, labels, targets, explanations, n_batches

    def _evaluate_dataset(
        self,
        eval_dataset: torch.utils.data.Dataset,
        explainer: Optional[Explainer],
        metric: Metric,
        batch_size: int,
        max_eval_n: Optional[int] = 1000,
        eval_seed: int = 42,
        precomputed_explanations: Optional[BatchedCachedExplanations] = None,
        inference_batch_size: Optional[int] = None,
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
        max_eval_n: Optional[int], optional
            Maximum number of evaluation samples to use. If None, uses the
            entire evaluation dataset. By default 1000.
        eval_seed: int, optional
            Random seed for evaluation sampling, by default 42.
        precomputed_explanations : Optional[BatchedCachedExplanations],
            optional
            If provided, these explanations will be used instead of computing
            them on the fly. By default None.
        inference_batch_size : Optional[int], optional
            Forwarded to :meth:`_iter_explanations` to sub-batch the model
            forward used for predictions. ``None`` keeps the full
            ``batch_size`` forward.

        Returns
        -------
        Any
            Computed metric result

        """
        for (
            _,
            inputs,
            labels,
            targets,
            explanations,
            _,
        ) in self._iter_explanations(
            explainer=explainer,
            eval_dataset=eval_dataset,
            batch_size=batch_size,
            max_eval_n=max_eval_n,
            eval_seed=eval_seed,
            precomputed_explanations=precomputed_explanations,
            inference_batch_size=inference_batch_size,
        ):
            data_unit = {
                "test_data": inputs,
                "test_targets": targets,
                "test_labels": labels,
                "explanations": explanations,
            }

            if hasattr(self, "entailment_labels"):
                data_unit["entailment_labels"] = self.entailment_labels

            if self.name == "Subclass Detection":
                data_unit["test_superclass_targets"] = torch.tensor(
                    [self.class_to_group[i.item()] for i in labels],  # type: ignore[attr-defined]
                    device=labels.device,
                )
                data_unit["test_targets"] = labels
                if not self.use_predictions:
                    data_unit["targets"] = data_unit["grouped_labels"]

            eval_unit = {k: data_unit[k] for k in self.eval_args}
            metric.update(**eval_unit)

        return metric.compute()
