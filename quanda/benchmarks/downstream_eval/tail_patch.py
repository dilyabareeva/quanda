"""Tail-Patch benchmark."""

import logging
from typing import Any, Callable, List, Optional, Type, Union

import datasets  # type: ignore
import torch
from torch.optim import Optimizer

from quanda.benchmarks.base import Benchmark
from quanda.metrics.downstream_eval.tail_patch import TailPatchMetric

logger = logging.getLogger(__name__)


class TailPatch(Benchmark):
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
        *args,
        **kwargs,
    ):
        """Initialize the Tail Patch benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: torch.nn.Module
        self.device: str
        self.train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset]
        self.eval_dataset: Union[torch.utils.data.Dataset, datasets.Dataset]
        self.checkpoints: List[str]
        self.checkpoints_load_func: Callable[..., Any]
        self.k: int
        self.learning_rate: float
        self.optimizer_class: Type[Optimizer]
        self.optimizer_kwargs: dict
        self.tokenizer_name: str

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
        obj.k = config.get("k", 10)
        obj.learning_rate = config.get("learning_rate", 1e-5)
        obj.optimizer_class = config.get("optimizer_class", torch.optim.AdamW)
        obj.optimizer_kwargs = config.get("optimizer_kwargs", {})
        obj.tokenizer_name = config.get("tokenizer_name", "gpt2")
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
            Batch size to be used for the evaluation, default to 8.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the evaluation results.

        """
        explainer = self._prepare_explainer(
            dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        metric = TailPatchMetric(
            model=self.model,
            train_dataset=self.train_dataset,
            checkpoints=self.checkpoints,
            k=self.k,
            learning_rate=self.learning_rate,
            optimizer_class=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            tokenizer_name=self.tokenizer_name,
        )

        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
