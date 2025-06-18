"""Benchmark for the Linear Datamodeling Score metric."""

import logging
import os
from typing import Any, Callable, List, Optional, Union

import lightning as L
import torch
import yaml
from huggingface_hub import create_repo, upload_folder

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

        obj.subset_ckpt_filenames = []
        for i in range(obj.m):
            obj.subset_ckpt_filenames.append(f"{config['repo_id']}/{config['ckpts'][-1]}_lds_subset_{i}")

        obj.alpha = config.get("alpha", 0.5)
        counterfactual_trainer = config.get(
            "counterfactual_trainer", config["model"].get("trainer", None)
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
        cache_dir = config.get("bench_save_dir", "./tmp")
        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=config, bench_save_dir=cache_dir
        )
        # create metadata dir if it doesn't exist
        os.makedirs(metadata_dir, exist_ok=True)

        generator = torch.Generator()
        generator.manual_seed(obj.seed)

        subset_meta = f"{metadata_dir}/{config['subset_ids']}"
        if (os.path.exists(subset_meta) and load_meta_from_disk):
            with open(f"{metadata_dir}/{config['subset_ids']}", "r") as f:
                obj.subset_ids = yaml.safe_load(f)
        else:
            obj.subset_ids = LinearDatamodelingMetric.generate_subsets(
                dataset=obj.train_dataset,
                alpha=obj.alpha,
                m=obj.m,
                generator=generator,
            )
            with open(f"{metadata_dir}/{config['subset_ids']}", "w") as f:
                f.write(f"{obj.subset_ids}")

        obj.model, obj.checkpoints, obj.checkpoints_load_func = (
            BenchConfigParser.parse_model_cfg(
                model_cfg=config["model"],
                bench_save_dir=config["bench_save_dir"],
                repo_id=config["repo_id"],
                ckpts=config["ckpts"],
                load_model_from_disk=offline,
                device=device,
            )
        )

        obj.correlation_fn = correlation_functions[config["correlation_fn"]]
        obj.use_predictions = config.get("use_predictions", True)
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
        obj = cls.from_config(config, load_meta_from_disk=False, offline=True, device=device)

        # Parse trainer configuration
        trainer = BenchConfigParser.parse_trainer_cfg(
            config["model"]["trainer"]
        )

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

        for i, filename in enumerate(obj.subset_ckpt_filenames):
            subset = torch.utils.data.Subset(obj.train_dataset, obj.subset_ids[i])
            subset_model = LinearDatamodelingMetric.train_subset_model(
                model=obj.model,
                subset=subset,
                trainer=trainer,
                batch_size=batch_size,
            )

            subset_model.push_to_hub(f"quanda-bench-test/{filename}")

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
            subset_ckpt_filenames=self.subset_ckpt_filenames,
            # TODO: implement pretrained_models in LDS metric
        )
        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
