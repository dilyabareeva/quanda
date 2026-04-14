"""Benchmark for the Linear Datamodeling Score metric."""

import logging
import os
import warnings
from copy import deepcopy
from typing import Callable, List, Optional

import lightning as L
import torch
import yaml

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.metrics.ground_truth.linear_datamodeling import (
    LinearDatamodelingMetric,
)
from quanda.utils.common import class_accuracy
from quanda.utils.functions import correlation_functions
from quanda.utils.training import Trainer

logger = logging.getLogger(__name__)


def _get_i_subset_ckpt_postfix(i: int) -> str:
    """Get checkpoint postfix for subset model i."""
    return f"_lds_subset_{i}"


def _get_i_subset_ckpt_name(ckpt_str: str, i: int) -> str:
    """Get full checkpoint name for subset model i."""
    return f"{ckpt_str}{_get_i_subset_ckpt_postfix(i)}"


class LinearDatamodeling(Benchmark):
    """Benchmark for the Linear Datamodeling Score metric.

    The LDS measures how well a data attribution method can predict the effect
    of retraining a model on different subsets of the training data. It
    computes the correlation between the model’s output when retrained on
    subsets of the data and the attribution method’s predictions of those
    outputs.

    References
    ----------
    1) Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc,
    and Aleksander Mądry. (2023). "TRAK: attributing model behavior at scale".
    In Proceedings of the 40th International Conference on Machine Learning"
    (ICML’23), Vol. 202. JMLR.org, Article 1128, (27074–27113).

    2) https://github.com/MadryLab/trak/

    """

    name: str = "Linear Datamodeling Score"
    eval_args: list = ["explanations", "test_data", "test_targets"]
    _push_subsets_during_train: bool = False
    _lds_skip_subsets: bool = False

    def __init__(
        self,
        *args,
        correlation_fn: Callable,
        m: int = 100,
        alpha: float = 0.5,
        cache_dir: str = "./tmp",
        model_id: str = "0",
        seed: int = 42,
        subset_ids: Optional[List[List[int]]] = None,
        subset_ckpt_filenames: Optional[List[str]] = None,
        counterfactual_trainer: Optional[Trainer] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize the `LinearDatamodeling` benchmark.

        Parameters
        ----------
        *args
            Positional arguments passed to the base class.
        m : int, optional
            Number of subsets, by default 100.
        alpha : float, optional
            Fraction of training data per subset, by default 0.5.
        cache_dir : str, optional
            Cache directory, by default "./tmp".
        model_id : str, optional
            Model identifier, by default "0".
        correlation_fn : Callable
            Correlation function to use.
        seed : int, optional
            Random seed, by default 42.
        subset_ids : Optional[List[List[int]]]
            Pre-computed subset indices.
        subset_ckpt_filenames : Optional[List[str]]
            Checkpoint filenames for subset models.
        counterfactual_trainer : Optional[Trainer]
            Trainer for counterfactual models.
        trainer_fit_kwargs : Optional[dict]
            Additional kwargs for trainer.fit().
        **kwargs
            Arguments passed to the base Benchmark class.

        """
        super().__init__(*args, **kwargs)
        self.m = m
        self.alpha = alpha
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.correlation_fn = correlation_fn
        self.seed = seed
        self.subset_ids = subset_ids or []
        self.subset_ckpt_filenames = subset_ckpt_filenames or []
        self.counterfactual_trainer = counterfactual_trainer
        self.trainer_fit_kwargs = trainer_fit_kwargs

    def _train_subset_model_by_idx(
        self,
        i: int,
        trainer: "Trainer",
        ckpt_str: str,
        ckpt_dir: str,
        batch_size: int = 8,
        push_to_hub: bool = False,
    ):
        """Train and save a single subset model by index."""
        subset = torch.utils.data.Subset(
            self.train_dataset, self.subset_ids[i]
        )
        subset_model = LinearDatamodelingMetric.train_subset_model(
            model=self.model,
            subset=subset,
            trainer=trainer,
            batch_size=batch_size,
        )

        local_ckpt_dir = f"{ckpt_dir}{_get_i_subset_ckpt_postfix(i)}"
        os.makedirs(local_ckpt_dir, exist_ok=True)
        if push_to_hub and len(os.listdir(local_ckpt_dir)) > 0:
            warnings.warn(
                f"Directory {local_ckpt_dir} already exists "
                "and is not empty. Checkpoints will be "
                "overwritten."
            )
        subset_model.save_pretrained(
            local_ckpt_dir, safe_serialization=True
        )

        if push_to_hub:
            subset_model.push_to_hub(
                _get_i_subset_ckpt_name(ckpt_str, i)
            )

    def _train_subset_models(
        self,
        trainer: "Trainer",
        ckpt_str: str,
        ckpt_dir: str,
        batch_size: int = 8,
        push_to_hub: bool = False,
    ):
        """Train and save all subset models."""
        for i in range(len(self.subset_ckpt_filenames)):
            self._train_subset_model_by_idx(
                i=i,
                trainer=trainer,
                ckpt_str=ckpt_str,
                ckpt_dir=ckpt_dir,
                batch_size=batch_size,
                push_to_hub=push_to_hub,
            )

    @classmethod
    def train(
        cls,
        config: dict,
        logger: Optional[L.pytorch.loggers.logger.Logger] = None,
        device: str = "cpu",
        batch_size: int = 64,
        skip_subsets: bool = False,
    ) -> "LinearDatamodeling":
        """Train main model and subset models.

        Extends the base train method to also train and save
        the counterfactual subset models required for LDS evaluation.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.
        logger : Optional[L.pytorch.loggers.logger.Logger], optional
            Logger to be used for logging, by default None.
        device : str, optional
            Device to use for training, by default "cpu".
        batch_size : int, optional
            Batch size for training, by default 8.
        skip_subsets : bool, optional
            If True, skip the subset training loop. Used when subsets
            are trained out-of-band (e.g. one-by-one in parallel
            workers via :meth:`train_subset`).

        Returns
        -------
        LinearDatamodeling
            The trained benchmark instance.

        """

        obj = super().train(
            config=config,
            logger=logger,
            device=device,
            batch_size=batch_size,
        )
        assert isinstance(obj, LinearDatamodeling)

        if skip_subsets or cls._lds_skip_subsets:
            return obj

        trainer = BenchConfigParser.parse_trainer_cfg(
            config["model"]["trainer"]
        )

        ckpt_dir = os.path.join(
            config.get("bench_save_dir", "./tmp"),
            "ckpt",
            config["ckpts"][-1].split("/")[-1],
        )

        obj._train_subset_models(
            trainer=trainer,
            ckpt_str=config["ckpts"][-1],
            ckpt_dir=ckpt_dir,
            batch_size=batch_size,
            push_to_hub=cls._push_subsets_during_train,
        )

        return obj

    @classmethod
    def train_subset(
        cls,
        config: dict,
        idx: int,
        device: str = "cpu",
        batch_size: int = 64,
        push_to_hub: bool = False,
    ) -> "LinearDatamodeling":
        """Train and save a single subset model by index.

        Builds the benchmark from existing on-disk metadata (the main
        model and subset_ids must already be available — typically
        produced by a prior call to :meth:`train` with
        ``skip_subsets=True``), then trains subset ``idx`` only.

        Parameters
        ----------
        config : dict
            Benchmark configuration dictionary.
        idx : int
            Subset index in ``[0, m)``.
        device : str, optional
            Device to train on.
        batch_size : int, optional
            Batch size.
        push_to_hub : bool, optional
            If True, push the resulting subset checkpoint to HF Hub.

        """
        obj = cls.from_config(
            config,
            load_meta_from_disk=True,
            offline=True,
            device=device,
        )
        assert isinstance(obj, LinearDatamodeling)

        trainer = BenchConfigParser.parse_trainer_cfg(
            config["model"]["trainer"]
        )
        ckpt_dir = os.path.join(
            config.get("bench_save_dir", "./tmp"),
            "ckpt",
            config["ckpts"][-1].split("/")[-1],
        )
        obj._train_subset_model_by_idx(
            i=idx,
            trainer=trainer,
            ckpt_str=config["ckpts"][-1],
            ckpt_dir=ckpt_dir,
            batch_size=batch_size,
            push_to_hub=push_to_hub,
        )
        return obj

    @classmethod
    def push_subset(
        cls,
        config: dict,
        idx: int,
    ) -> None:
        """Push an already-trained subset checkpoint to HF Hub.

        Used to serialize Hub uploads after parallel local training.
        """
        from huggingface_hub import HfApi  # local import; optional dep path

        ckpt_str = config["ckpts"][-1]
        ckpt_dir = os.path.join(
            config.get("bench_save_dir", "./tmp"),
            "ckpt",
            ckpt_str.split("/")[-1],
        )
        local_ckpt_dir = f"{ckpt_dir}{_get_i_subset_ckpt_postfix(idx)}"
        repo_id = _get_i_subset_ckpt_name(ckpt_str, idx)

        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(folder_path=local_ckpt_dir, repo_id=repo_id)

    @classmethod
    def _extra_kwargs_from_config(
        cls,
        config: dict,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        metadata_dir: str,
        load_meta_from_disk: bool,
    ) -> dict:
        """Extract linear datamodeling kwargs from config."""
        m = config.get("m", 100)
        alpha = config.get("alpha", 0.5)
        seed = config["seed"]

        ckpt = config["ckpts"][-1]

        subset_ckpt_filenames = [
            _get_i_subset_ckpt_name(ckpt, i) for i in range(m)
        ]
        counterfactual_trainer_cfg = config.get(
            "counterfactual_trainer",
            config["model"].get("trainer", None),
        )
        if counterfactual_trainer_cfg is None:
            raise ValueError(
                "Either ‘trainer’ or ‘model.trainer’ should be set."
            )
        counterfactual_trainer = BenchConfigParser.parse_trainer_cfg(
            counterfactual_trainer_cfg
        )

        os.makedirs(metadata_dir, exist_ok=True)

        generator = torch.Generator()
        generator.manual_seed(seed)

        subset_meta = f"{metadata_dir}/{config['subset_ids']}"
        if os.path.exists(subset_meta) and load_meta_from_disk:
            with open(subset_meta, "r") as f:
                subset_ids = yaml.safe_load(f)
        else:
            subset_ids = LinearDatamodelingMetric.generate_subsets(
                dataset=train_dataset,
                alpha=alpha,
                m=m,
                generator=generator,
            )
            with open(subset_meta, "w") as f:
                f.write(f"{subset_ids}")

        return {
            "m": m,
            "alpha": alpha,
            "cache_dir": config.get("cache_dir", "./tmp"),
            "model_id": config.get("model_id", "0"),
            "correlation_fn": correlation_functions[config["correlation_fn"]],
            "seed": seed,
            "subset_ids": subset_ids,
            "subset_ckpt_filenames": subset_ckpt_filenames,
            "counterfactual_trainer": counterfactual_trainer,
            # TODO: make trainer_fit_kwargs available to all benchmarks
            "trainer_fit_kwargs": config.get("trainer_fit_kwargs", None),
        }

    @classmethod
    def train_and_push_to_hub(
        cls,
        config: dict,
        logger: Optional[L.pytorch.loggers.logger.Logger] = None,
        device: str = "cpu",
        batch_size: int = 64,
    ):  # pragma: no cover
        """Train a model using the provided config and push to HF hub."""
        skip_subsets = bool(config.get("skip_subsets", False))
        cls._push_subsets_during_train = not skip_subsets
        cls._lds_skip_subsets = skip_subsets
        try:
            obj = super().train_and_push_to_hub(
                config=config,
                logger=logger,
                device=device,
                batch_size=batch_size,
            )
        finally:
            cls._push_subsets_during_train = False
            cls._lds_skip_subsets = False
        assert isinstance(obj, LinearDatamodeling)
        return obj

    def sanity_check(self, batch_size: int = 32) -> dict:
        """Compute accuracy of main model and all subset checkpoints.

        Parameters
        ----------
        batch_size : int, optional
            Batch size to be used for the evaluation, defaults to 32.

        Returns
        -------
        dict
            Dictionary containing the sanity check results, including
            per-subset checkpoint accuracies.

        """
        results = super().sanity_check(batch_size)

        eval_dl = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        for i, ckpt_path in enumerate(self.subset_ckpt_filenames):
            subset_model = deepcopy(self.model)
            self.checkpoints_load_func(subset_model, ckpt_path)
            subset_model.eval()
            subset_model.to(self.device)
            acc = class_accuracy(subset_model, eval_dl, self.device)
            results[f"subset_acc_{i}"] = acc

        return results

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
        precomputed = self._resolve_precomputed_explanations(
            cache_dir=cache_dir,
            use_cached_expl=use_cached_expl,
            use_hf_expl=use_hf_expl,
        )
        explainer = (
            None
            if precomputed is not None
            else self._prepare_explainer(
                dataset=self.train_dataset,
                explainer_cls=explainer_cls,
                expl_kwargs=expl_kwargs,
            )
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
        )
        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
            max_eval_n=max_eval_n,
            eval_seed=eval_seed,
            precomputed_explanations=precomputed,
        )
