"""Recall@k benchmark."""

import logging
from typing import Any, Callable, List, Optional, Union

import datasets  # type: ignore
import torch

from quanda.benchmarks.downstream_eval._fact_tracing import (
    FactTracingBenchmark,
)
from quanda.metrics.downstream_eval.recall_at_k import RecallAtKMetric

logger = logging.getLogger(__name__)


class RecallAtK(FactTracingBenchmark):
    """Benchmark for Recall@k metric.

    This benchmark evaluates whether retrieved examples (proponents) logically
    support or entail a given fact by measuring the proportion of facts for
    which an entailing proponent appears in the top k proponent retrievals.

    References
    ----------
    1) Tyler A. Chang, Dheeraj Rajagopal, Tolga Bolukbasi, Lucas Dixon,
    and Ian Tenney. (2024) "Scalable Influence and Fact Tracing for
    Large Language Model Pretraining". The Thirteenth International
    Conference on Learning Representations.

    """

    name: str = "Recall@k"

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
        entailment_labels: Optional[torch.Tensor] = None,
        k: int = 10,
    ):
        """Initialize the Recall@k benchmark.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to produce attributions.
        train_dataset : torch.utils.data.Dataset or datasets.Dataset
            Training dataset (proponent pool).
        eval_dataset : torch.utils.data.Dataset
            Evaluation dataset of facts.
        checkpoints : list of str
            Paths to model checkpoints.
        checkpoints_load_func : Callable
            Function used to load each checkpoint into ``model``.
        device : str, optional
            Torch device, by default ``"cpu"``.
        val_dataset : torch.utils.data.Dataset or datasets.Dataset, optional
            Optional validation dataset, by default ``None``.
        use_predictions : bool, optional
            Whether to attribute model predictions instead of labels, by
            default ``False``.
        entailment_labels : torch.Tensor, optional
            Binary ``(n_eval, n_train)`` fact entailment matrix, by default
            ``None``.
        k : int, optional
            The k value for Recall@k, by default 10.

        """
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            device=device,
            val_dataset=val_dataset,
            use_predictions=use_predictions,
            entailment_labels=entailment_labels,
        )
        self.k: int = k

    @classmethod
    def _extra_kwargs_from_config(
        cls,
        *args,
        **kwargs,
    ) -> dict:
        """Pull ``k`` off the config into ``__init__`` kwargs."""
        config = args[0] if args else kwargs
        return {"k": config.get("k", 10)}

    def _build_metric(self, inference_batch_size=None) -> RecallAtKMetric:
        """Instantiate the Recall@k metric."""
        return RecallAtKMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.train_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            k=self.k,
        )
