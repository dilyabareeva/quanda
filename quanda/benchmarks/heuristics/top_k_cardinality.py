"""Top-K Cardinality benchmark module."""

import logging
from typing import Optional, Union

import datasets  # type: ignore
import torch

from quanda.benchmarks.base import Benchmark
from quanda.metrics.heuristics import TopKCardinalityMetric

logger = logging.getLogger(__name__)


class TopKCardinality(Benchmark):
    # TODO: remove USES PREDICTED LABELS https://arxiv.org/pdf/2006.04528
    """Benchmark for the Top-K Cardinality heuristic.

    This benchmark evaluates the dependence of the attributions on the test
    samples being attributed. The cardinality of the union of top-k attributed
    training samples is computed. A higher cardinality indicates variance in
    the attributions, which indicates dependence on the test samples.

    References
    ----------
    1) Barshan, Elnaz, Marc-Etienne Brunet, and Gintare Karolina Dziugaite.
    (2020). Relatif: Identifying explanatory training samples via relative
    influence. International Conference on Artificial Intelligence and
    Statistics. PMLR.

    """

    name: str = "Top-K Cardinality"
    eval_args: list = ["explanations"]
    default_use_predictions: bool = True

    def __init__(
        self,
        *args,
        top_k: int = 5,
        **kwargs,
    ):
        """Initialize the Top-K Cardinality benchmark.

        Parameters
        ----------
        *args
            Positional arguments passed to the base Benchmark class.
        top_k : int, optional
            Number of top attributed samples to consider, by default 5.
        **kwargs
            Arguments passed to the base Benchmark class.

        """
        super().__init__(*args, **kwargs)
        self.top_k = top_k

    @classmethod
    def _extra_kwargs_from_config(
        cls,
        config: dict,
        train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        metadata_dir: str,
        load_meta_from_disk: bool,
    ) -> dict:
        """Extract top_k from config."""
        return {"top_k": config["top_k"]}

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
        """Evaluate the benchmark using a given explanation method.

        Parameters
        ----------
        explainer_cls: type
            The explanation class inheriting from the base Explainer class to
            be used for evaluation.
        expl_kwargs: Optional[dict], optional
            Keyword arguments for the explainer, by default None.
        batch_size: int, optional
            Batch size for the evaluation, by default 8.
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
        inference_batch_size: Optional[int], optional
            If set, split the per-batch model forward used for predictions
            into sub-batches of this size. ``None`` keeps the full
            ``batch_size`` forward.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.

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

        metric = TopKCardinalityMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.train_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            top_k=self.top_k,
        )

        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
            max_eval_n=max_eval_n,
            eval_seed=eval_seed,
            precomputed_explanations=precomputed,
            inference_batch_size=inference_batch_size,
        )
