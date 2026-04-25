"""Tail-Patch benchmark."""

import logging
from typing import Any, Callable, List, Optional, Type, Union

import datasets  # type: ignore
import torch
from torch.optim import Optimizer

from quanda.benchmarks.downstream_eval._fact_tracing import (
    FactTracingBenchmark,
)
from quanda.metrics.downstream_eval.tail_patch import TailPatchMetric

logger = logging.getLogger(__name__)


class TailPatch(FactTracingBenchmark):
    """Benchmark for Tail Patch metric.

    This benchmark evaluates the effectiveness of a training data attribution
    method by estimating how much its top-k retrieved training examples
    increase the model's probability of predicting a target output when trained
    on each of them individually.

    References
    ----------
    1) Tyler A. Chang, Dheeraj Rajagopal, Tolga Bolukbasi, Lucas Dixon,
    and Ian Tenney. (2024) "Scalable Influence and Fact Tracing for
    Large Language Model Pretraining". The Thirteenth International
    Conference on Learning Representations.

    """

    name: str = "Tail Patch"
    eval_args: list = ["explanations", "test_data", "test_targets"]

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
        learning_rate: float = 1e-5,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[dict] = None,
        tokenizer_name: str = "gpt2",
    ):
        """Initialize the Tail Patch benchmark.

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
            Number of top proponents to evaluate, by default 10.
        learning_rate : float, optional
            Learning rate for the single gradient step, by default 1e-5.
        optimizer_class : Type[Optimizer], optional
            Optimizer class, by default ``torch.optim.AdamW``.
        optimizer_kwargs : Optional[dict], optional
            Extra optimizer kwargs.
        tokenizer_name : str, optional
            HF tokenizer name used inside
            :class:`~quanda.metrics.downstream_eval.tail_patch.TailPatchMetric`.

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
        self.learning_rate: float = learning_rate
        self.optimizer_class: Type[Optimizer] = optimizer_class
        self.optimizer_kwargs: dict = optimizer_kwargs or {}
        self.tokenizer_name: str = tokenizer_name

    @classmethod
    def _extra_kwargs_from_config(cls, config: dict) -> dict:
        """Pull TailPatch-specific kwargs off the config."""
        return {
            "k": config.get("k", 10),
            "learning_rate": config.get("learning_rate", 1e-5),
            "optimizer_class": config.get(
                "optimizer_class", torch.optim.AdamW
            ),
            "optimizer_kwargs": config.get("optimizer_kwargs", {}),
            "tokenizer_name": config.get("tokenizer_name", "gpt2"),
        }

    def _build_metric(
        self, inference_batch_size: Optional[int] = None
    ) -> TailPatchMetric:
        """Instantiate the Tail-Patch metric."""
        return TailPatchMetric(
            model=self.model,
            train_dataset=self.train_dataset,
            checkpoints=self.checkpoints,
            k=self.k,
            learning_rate=self.learning_rate,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            tokenizer_name=self.tokenizer_name,
            inference_batch_size=inference_batch_size,
        )
