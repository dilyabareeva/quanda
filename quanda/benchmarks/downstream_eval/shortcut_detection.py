"""Shortcut Detection Benchmark."""

from typing import List, Optional

import torch
from torch.utils.data import Subset

from quanda.benchmarks.base import Benchmark
from quanda.metrics.downstream_eval.shortcut_detection import (
    ShortcutDetectionMetric,
)
from quanda.utils.common import (
    DatasetSplit,
    class_accuracy,
    ds_len,
)
from quanda.utils.datasets.transformed.sample import (
    SampleTransformationDataset,
)


class ShortcutDetection(Benchmark):
    # TODO: Add citation to the original paper formulating ShortcutDetection
    #  after acceptance
    """Benchmark for shortcut detection evaluation task.

    A class is selected, and a subset of its images is modified by overlaying a
    shortcut trigger. The model is then trained on this dataset and learns to
    use the shortcut as a trigger to predict the class. The objective is to
    detect this shortcut by analyzing the model's attributions.

    Note that all explanations are generated with respect to the class of the
    shortcut samples, to detect the shortcut.

    The average attributions for triggered examples from the class, clean
    examples from the class, and clean examples from other classes are
    computed.

    This metric is inspired by the Domain Mismatch Detection Test of Koh et al.
    (2017) and the Backdoor Poisoning Detection.

    References
    ----------
    1) Koh, Pang Wei, and Percy Liang. (2017). Understanding black-box
    predictions via influence functions. International conference on machine
    learning. PMLR.

    """

    name: str = "Shortcut Detection"
    eval_args = ["test_data", "test_labels", "explanations"]

    def __init__(
        self,
        *args,
        shortcut_cls: int = 0,
        filter_by_non_shortcut: bool = False,
        filter_by_shortcut_pred: bool = False,
        filter_indices: Optional[List[int]] = None,
        **kwargs,
    ):
        """Initialize the benchmark object.

        Parameters
        ----------
        *args
            Positional arguments passed to the base class.
        shortcut_cls : int
            The class index used as the shortcut target.
        filter_by_non_shortcut : bool
            Whether to filter the test samples to only calculate the metric on
            those samples, where the shortcut class
            is not assigned as the class, by default True.
        filter_by_shortcut_pred: bool
            Whether to filter the test samples to only calculate the metric on
            those samples, where the shortcut class
            is predicted, by default True.
        filter_indices : Optional[List[int]], optional
            Pre-computed indices for filtering eval samples, by default
            None.
        **kwargs
            Arguments passed to the base Benchmark class.

        """
        super().__init__(*args, **kwargs)
        self.shortcut_cls = shortcut_cls
        self.filter_by_non_shortcut = filter_by_non_shortcut
        self.filter_by_shortcut_pred = filter_by_shortcut_pred
        self.filter_indices = filter_indices

    @classmethod
    def _extra_kwargs_from_config(
        cls,
        config: dict,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: torch.utils.data.Dataset,
        metadata_dir: str,
        load_meta_from_disk: bool,
    ) -> dict:
        """Extract shortcut detection kwargs from config."""
        if not isinstance(eval_dataset, SampleTransformationDataset):
            raise ValueError(
                "Shortcut detection evaluation requires a "
                "SampleTransformationDataset as the evaluation dataset."
            )
        if not isinstance(train_dataset, SampleTransformationDataset):
            raise ValueError(
                "Shortcut detection evaluation requires a "
                "SampleTransformationDataset as the training dataset."
            )

        assert train_dataset.metadata.cls_idx is not None, (
            "The training dataset must have a class index in its metadata."
        )

        eval_ds_config = config["eval_dataset"]
        eval_indices = eval_ds_config["filter_indices"]
        filter_indices = None

        if (
            DatasetSplit.exists(metadata_dir, eval_indices["split_filename"])
            and load_meta_from_disk
        ):
            filter_indices = DatasetSplit.load(
                metadata_dir,
                name=eval_indices["split_filename"],
            )[eval_indices["split_name"]]

        return {
            "shortcut_cls": train_dataset.metadata.cls_idx,
            "filter_by_non_shortcut": config.get(
                "filter_by_non_shortcut", False
            ),
            "filter_by_shortcut_pred": config.get(
                "filter_by_shortcut_pred", False
            ),
            "filter_indices": filter_indices,
        }

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

        assert isinstance(self.train_dataset, SampleTransformationDataset)
        assert isinstance(self.eval_dataset, SampleTransformationDataset)

        train_dl = torch.utils.data.DataLoader(
            Subset(self.train_dataset, self.train_dataset.transform_indices),
            batch_size=batch_size,
            shuffle=False,
        )

        eval_dl = torch.utils.data.DataLoader(
            Subset(self.eval_dataset, self.eval_dataset.transform_indices),
            batch_size=batch_size,
            shuffle=False,
        )

        results["train_shortcut_memorization"] = class_accuracy(
            self.model, train_dl, self.device
        )
        results["eval_shortcut_memorization"] = class_accuracy(
            self.model,
            eval_dl,
            single_class=self.shortcut_cls,
            device=self.device,
        )
        # .dataset is the Subset created by apply_filter;
        # .dataset.dataset is the pre-filter eval split.
        if hasattr(self.eval_dataset, "dataset") and hasattr(
            self.eval_dataset.dataset, "dataset"
        ):
            results["eval_post_filter_percentage"] = ds_len(
                self.eval_dataset
            ) / ds_len(self.eval_dataset.dataset.dataset)
        return results

    def overall_objective(self, sanity_check_results: dict) -> float:
        """Compute overall objective score.

        Based on sanity check results, for selecting optional
        hyperparameters of the benchmark.
        Assigns extra weight to the eval_post_filter_percentage,
        as it is the most direct indicator of whether the model
        has learned the shortcut.

        Parameters
        ----------
        sanity_check_results : dict
            Dictionary containing the results from the sanity check.

        Returns
        -------
        float
            Overall objective score computed from the sanity check results.

        """
        train_acc = sanity_check_results.get("train_acc", 0)
        val_acc = sanity_check_results.get("val_acc", 0)
        train_shortcut_memorization = sanity_check_results.get(
            "train_shortcut_memorization", 0
        )
        eval_post_filter_percentage = sanity_check_results.get(
            "eval_post_filter_percentage", 0
        )
        return (
            0.1 * train_acc
            + 0.2 * val_acc
            + 0.1 * train_shortcut_memorization
            + 0.6 * eval_post_filter_percentage
        )

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

        assert isinstance(self.train_dataset, SampleTransformationDataset)
        metric = ShortcutDetectionMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
            train_dataset=self.train_dataset,
            shortcut_indices=self.train_dataset.transform_indices,
            shortcut_cls=self.shortcut_cls,
            filter_by_non_shortcut=self.filter_by_non_shortcut,
            filter_by_shortcut_pred=self.filter_by_shortcut_pred,
        )
        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )

    def _compute_and_save_indices(self, config: dict, batch_size: int = 8):
        """Run inference on eval_dataset and save the filter indices.

        Iterates over ``self.eval_dataset``, calls ``_compute_filter_mask``
        on every batch, collects the selected indices, stores them in
        ``self.filter_indices``, and persists them via
        ``save_filtered_indices``.

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
            filter_by_shortcut_pred=self.filter_by_shortcut_pred,
            shortcut_cls=self.shortcut_cls,
            filter_by_non_shortcut=self.filter_by_non_shortcut,
        )
