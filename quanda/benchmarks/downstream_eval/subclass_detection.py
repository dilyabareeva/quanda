"""Benchmark for subclass detection task."""

import logging
from typing import Dict, Optional

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
    eval_args = ["test_labels", "explanations", "test_data", "test_targets"]

    def __init__(
        self,
        *args,
        class_to_group: Optional[Dict[int, int]] = None,
        filter_by_prediction: bool = False,
        **kwargs,
    ):
        """Initialize the Subclass Detection benchmark.

        Parameters
        ----------
        *args
            Positional arguments passed to the base class.
        class_to_group : Optional[Dict[int, int]]
            Mapping from class index to group index.
        filter_by_prediction : bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the correct superclass is predicted, by
            default False.
        **kwargs
            Arguments passed to the base Benchmark class.

        """
        super().__init__(*args, **kwargs)
        self.class_to_group = class_to_group
        self.filter_by_prediction = filter_by_prediction

        # Ensure all datasets use the same class_to_group mapping.
        if class_to_group is not None:
            for ds in (
                self.train_dataset,
                self.eval_dataset,
                self.val_dataset,
            ):
                if isinstance(ds, LabelGroupingDataset):
                    ds.class_to_group = class_to_group

    @classmethod
    def _extra_kwargs_from_config(
        cls,
        config: dict,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        metadata_dir: str,
        load_meta_from_disk: bool,
    ) -> dict:
        """Extract subclass detection kwargs from config."""
        if not isinstance(train_dataset, LabelGroupingDataset):
            raise ValueError(
                "The train dataset must be a LabelGroupingDataset."
            )

        return {
            "class_to_group": train_dataset.class_to_group,
            "filter_by_prediction": config.get("filter_by_prediction", False),
        }

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

    def _compute_and_save_indices(self, config: dict, batch_size: int = 8):
        """Determine the indices of eval dataset.

        Filter by labels and predictions. By default,
        all samples are kept.

        Parameters
        ----------
        config : dict
            Benchmark configuration dictionary (needed for save path).
        batch_size : int, optional
            Batch size for the inference pass, by default 8.

        """
        super()._compute_and_save_filter_by_labels_and_prediction(
            config=config,
            batch_size=batch_size,
            filter_by_prediction=self.filter_by_prediction,
        )

    def sanity_check(self, batch_size: int = 32) -> dict:
        """Compute accuracy on shortcut datapoints as a sanity check.

        Parameters
        ----------
        batch_size : int, optional
            Batch size to be used for the evaluation, default to 32.

        Returns
        -------
        dict
            Dictionary containing the evaluation results.

        """
        results = super().sanity_check(batch_size)

        # .dataset is the Subset created by apply_filter;
        # .dataset.dataset is the pre-filter eval split.
        if hasattr(self.eval_dataset, "dataset") and hasattr(
            self.eval_dataset.dataset, "dataset"
        ):
            results["eval_post_filter_percentage"] = ds_len(
                self.eval_dataset
            ) / ds_len(self.eval_dataset.dataset.dataset)
        return results
