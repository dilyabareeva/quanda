"""Top-K Cardinality benchmark module."""

import logging
from typing import Callable, List, Optional, Any

import torch
import torch.utils

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

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the Top-K Cardinality benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: torch.nn.Module
        self.device: str
        self.train_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.use_predictions: bool
        self.checkpoints: List[str]
        self.checkpoints_load_func: Callable[..., Any]
        self.top_k: int

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
        obj.top_k = config["top_k"]
        obj.use_predictions = config.get("use_predictions", True)
        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
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

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.

        """
        explainer = self._prepare_explainer(
            dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
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
        )
