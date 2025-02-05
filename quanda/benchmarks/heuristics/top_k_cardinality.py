"""Top-K Cardinality benchmark module."""

import logging
from typing import Callable, List, Optional, Union, Any

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

        self.train_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.use_predictions: bool
        self.top_k: int

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        dataset_transform: Optional[Callable] = None,
        top_k: int = 1,
        use_predictions: bool = True,
        dataset_split: str = "train",
        checkpoint_paths: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """Assembles the benchmark from existing components.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be evaluated.
        train_dataset : Union[str, torch.utils.data.Dataset]
            The training dataset used to train the model.
        eval_dataset : torch.utils.data.Dataset
            The dataset to be used for the evaluation.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        dataset_transform : Optional[Callable], optional
            The transform to be applied to the dataset, by default None.
        top_k : int, optional
            The number of top-k samples to consider, by default 1.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation, by
            default True.
        dataset_split : str, optional
            The dataset split, by default "train", only used for HuggingFace
            datasets.
        checkpoint_paths : Optional[List[str]], optional
            List of paths to the checkpoints. This parameter is only used for
            downloaded benchmarks, by default None.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        TopKCardinality
            The benchmark instance.

        """
        obj = cls()
        obj._assemble_common(
            model=model,
            eval_dataset=eval_dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            use_predictions=use_predictions,
        )

        obj.train_dataset = obj._process_dataset(
            train_dataset,
            transform=dataset_transform,
            dataset_split=dataset_split,
        )
        obj.top_k = top_k
        obj._checkpoint_paths = checkpoint_paths

        return obj

    generate = assemble

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
