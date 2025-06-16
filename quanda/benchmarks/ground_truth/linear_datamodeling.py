"""Benchmark for the Linear Datamodeling Score metric."""

import logging
import os
from typing import Any, Callable, List, Optional, Union

import lightning as L
import torch

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.metrics.ground_truth.linear_datamodeling import (
    LinearDatamodelingMetric,
)
from quanda.utils.functions import correlation_functions
from quanda.utils.training import BaseTrainer

logger = logging.getLogger(__name__)


class LinearDatamodeling(Benchmark):
    """Benchmark for the Linear Datamodeling Score metric.

    The LDS measures how well a data attribution method can predict the effect
    of retraining a model on different subsets of the training data. It
    computes the correlation between the model’s output when retrained on
    subsets of the data and the attribution method's predictions of those
    outputs.

    References
    ----------
    1) Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc,
    and Aleksander Mądry. (2023). "TRAK: attributing model behavior at scale".
    In Proceedings of the 40th International Conference on Machine Learning"
    (ICML'23), Vol. 202. JMLR.org, Article 1128, (27074–27113).

    2) https://github.com/MadryLab/trak/

    """

    name: str = "Linear Datamodeling Score"
    eval_args: list = ["explanations", "test_data", "test_targets"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the `LinearDatamodeling` benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]
        self.device: str
        self.train_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.dataset_transform: Optional[Callable]
        self.m: int
        self.alpha: float
        self.counterfactual_trainer: Union[L.Trainer, BaseTrainer]
        self.trainer_fit_kwargs: Optional[dict]
        self.cache_dir: str
        self.model_id: str
        self.use_predictions: bool
        self.correlation_fn: Callable
        self.seed: int
        self.subset_ids: Optional[List[List[int]]]
        self.pretrained_models: Optional[List[str]]

        self.checkpoints: List[str]
        self.checkpoints_load_func: Callable[..., Any]

    @classmethod
    def from_config(
        cls,
        config: dict,
        load_meta_from_disk: bool = True,
        offline: bool = False,
        device: str = "cpu",
    ):
        """Initialize the benchmark from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.
        load_meta_from_disk : str
            Loads dataset metadata from disk if True, otherwise generates it,
            default True.
        offline : bool
            If True, the model is not downloaded, default False.
        device: str, optional
            Device to use for the evaluation, by default "cpu".

        """
        obj = super().from_config(config, load_meta_from_disk, offline, device)
        obj.m = config.get("m", 100)
        obj.alpha = config.get("alpha", 0.5)
        counterfactual_trainer = config.get(
            "counterfactual_trainer",
            config["model"].get("trainer", None)
        )
        if counterfactual_trainer is None:
            raise ValueError(
                "Either 'trainer' or 'model.trainer' should be set."
            )
        obj.counterfactual_trainer = BenchConfigParser.parse_trainer_cfg(
            counterfactual_trainer
        )
        # TODO: make trainer_fit_kwargs available to all benchmarks
        obj.trainer_fit_kwargs = config.get("trainer_fit_kwargs", None)
        obj.model_id = config.get("model_id", "0")
        obj.cache_dir = config.get("cache_dir", "./tmp")
        obj.seed = config["seed"]

        metadata_dir = BenchConfigParser.load_metadata(
            cfg=config,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )

        # create metadata dir if it doesn't exist
        os.makedirs(metadata_dir, exist_ok=True)
        subset_ids = f"{metadata_dir}/{config['subset_ids']}"
        if not os.path.exists(subset_ids):
            subset_ids = None

        obj.generator = torch.Generator()
        obj.generator.manual_seed(obj.seed)
        obj.subset_ids = LinearDatamodelingMetric.generate_subsets(
            dataset=obj.train_dataset,
            subset_ids=subset_ids,
            alpha=obj.alpha,
            m=obj.m,
            cache_dir=metadata_dir,
            generator=obj.generator,
            device=device,
        )

        with open(f"{metadata_dir}/{config['subset_ids']}", "w") as f:
            f.write(f"{obj.subset_ids}")

        obj.model, obj.checkpoints, obj.checkpoints_load_func = (
            BenchConfigParser.parse_model_cfg(
                model_cfg=config["model"],
                bench_save_dir=config["bench_save_dir"],
                repo_id=config["repo_id"],
                ckpts=config["ckpts"],
                offline=offline,
                device=device,
            )
        )

        _, obj.pretrained_models, _ = (
            BenchConfigParser.parse_model_cfg(
                model_cfg=config["model"],
                bench_save_dir=config["bench_save_dir"],
                repo_id=config["repo_id"],
                ckpts=config["subset_ckpts"],
                offline=offline,
                device=device,
            )
        )

        obj.correlation_fn = correlation_functions[config["correlation_fn"]]
        obj.use_predictions = config.get("use_predictions", True)
        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        """Evaluate the given data attributor.

        Parameters
        ----------
        explainer_cls : type
            Class of the explainer to be used for the evaluation.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer, by default None.
        batch_size : int, optional
            Batch size to be used for the evaluation, defaults to 8

        Returns
        -------
        dict
            Dictionary containing the evaluation results.

        """
        explainer = self._prepare_explainer(
            dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        metric = LinearDatamodelingMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
            train_dataset=self.train_dataset,
            alpha=self.alpha,
            m=self.m,
            cache_dir=self.cache_dir,
            model_id=self.model_id,
            correlation_fn=self.correlation_fn,
            seed=self.seed,
            batch_size=batch_size,
            subset_ids=self.subset_ids,
            pretrained_models=self.pretrained_models,
            # TODO: implement pretrained_models in LDS metric
        )
        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
