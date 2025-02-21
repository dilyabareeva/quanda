"""Benchmark for subclass detection task."""

import logging
from typing import Callable, Dict, List, Optional, Union, Any

import lightning as L
import torch

from quanda.benchmarks.base import Benchmark
from quanda.metrics.downstream_eval import SubclassDetectionMetric
from quanda.utils.common import ds_len
from quanda.utils.datasets.transformed.label_grouping import (
    LabelGroupingDataset,
)

logger = logging.getLogger(__name__)


class SubclassDetection(Benchmark):
    """Benchmark for subclass detection task.

    A model is trained on a dataset where labels are grouped into superclasses.
    The metric evaluates the performance of an attribution method in detecting
    the subclass of a test sample from its highest attributed training point.

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of
    similarity-based explanations. In International Conference on Learning
    Representations.

    """

    name: str = "Subclass Detection"
    eval_args = ["test_labels", "explanations", "test_data", "grouped_labels"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the Subclass Detection benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]
        self.train_dataset: LabelGroupingDataset
        self.eval_dataset: LabelGroupingDataset
        self.class_to_group: Dict[int, int]
        self.device: str

        self.use_predictions: bool
        self.filter_by_prediction: bool
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
        obj.class_to_group = obj.train_dataset.class_to_group
        obj.filter_by_prediction = config.get("filter_by_prediction", False)
        obj.use_predictions = config.get("use_predictions", True)
        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        *args,
        **kwargs,
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
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.

        """
        if not isinstance(self.train_dataset, LabelGroupingDataset):
            raise ValueError(
                "The train dataset must be a LabelGroupingDataset."
            )

        if not isinstance(self.eval_dataset, LabelGroupingDataset):
            raise ValueError(
                "The eval dataset must be a LabelGroupingDataset."
            )

        explainer = self._prepare_explainer(
            dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        metric = SubclassDetectionMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.train_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            train_subclass_labels=torch.tensor(
                [
                    self.train_dataset.dataset[s][1]
                    for s in range(ds_len(self.train_dataset.dataset))
                ]
            ),
            filter_by_prediction=self.filter_by_prediction,
        )

        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
