"""Model Randomization benchmark module."""

import logging
from typing import Callable, Optional

import torch

from quanda.benchmarks.base import Benchmark
from quanda.metrics.heuristics.model_randomization import (
    ModelRandomizationMetric,
)
from quanda.utils.functions import correlation_functions

logger = logging.getLogger(__name__)


class ModelRandomization(Benchmark):
    """Benchmark for the model randomization heuristic.

    This benchmark is used to evaluate the dependence of the attributions on
    the model parameters.

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of
    similarity-based explanations. In International Conference on Learning
    Representations.

    2) Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim,
    B. (2018). Sanity checks for saliency maps. In Advances in Neural
    Information Processing Systems (Vol. 31).

    """

    name: str = "Model Randomization"
    eval_args: list = ["explanations", "test_data", "test_targets"]
    default_use_predictions: bool = False

    def __init__(
        self,
        *args,
        correlation_fn: Callable,
        model_id: str = "0",
        cache_dir: str = "./tmp",
        seed: int = 42,
        **kwargs,
    ):
        """Initialize the Model Randomization benchmark.

        Parameters
        ----------
        *args
            Positional arguments passed to the base class.
        correlation_fn : Callable
            Correlation function to use.
        model_id : str, optional
            Model identifier, by default "0".
        cache_dir : str, optional
            Cache directory, by default "./tmp".
        seed : int, optional
            Random seed, by default 42.
        **kwargs
            Arguments passed to the base Benchmark class.

        """
        super().__init__(*args, **kwargs)
        self.correlation_fn = correlation_fn
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.seed = seed

    @classmethod
    def _extra_kwargs_from_config(
        cls,
        config: dict,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        metadata_dir: str,
        load_meta_from_disk: bool,
    ) -> dict:
        """Extract model randomization kwargs from config."""
        return {
            "correlation_fn": correlation_functions[config["correlation_fn"]],
            "model_id": config.get("model_id", "0"),
            "cache_dir": config.get("bench_save_dir", "./tmp"),
            "seed": config["seed"],
        }

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
            Batch size to be used for the evaluation, default to 8.
        max_eval_n: Optional[int], optional
            Maximum number of evaluation samples to use. If None, uses the
            entire evaluation dataset. By default 1000.
        eval_seed: int, optional
            Random seed for evaluation sampling, by default 42.
        cache_dir: Optional[str], optional
            Directory where cached explanations are stored. Required if
            `use_cached_expl` or `use_hf_expl` is True. By default None.
        use_cached_expl: bool, optional
            Whether to use cached explanations, by default False.
        use_hf_expl: bool, optional
            Whether to use Hugging Face cached explanations, by default False.
            If use_cached_expl is also True, will prioritize local cache over
            HF cache.

        Returns
        -------
        Dict[str, float]
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

        metric = ModelRandomizationMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
            model_id=self.model_id,
            cache_dir=self.cache_dir,
            train_dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            correlation_fn=self.correlation_fn,
            seed=self.seed,
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
