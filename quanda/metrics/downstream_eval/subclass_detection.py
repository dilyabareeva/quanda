"""Subclass Detection Metric."""

from typing import List, Optional, Union, Callable, Any

import torch

from quanda.metrics.downstream_eval import ClassDetectionMetric
from quanda.utils.common import ds_len


class SubclassDetectionMetric(ClassDetectionMetric):
    """Subclass Detection Metric as defined in Hanawa et al. (2021).

    A model is trained on a dataset where labels are grouped into superclasses.
    The metric evaluates the performance of an attribution method in detecting
    the subclass of a test sample from its highest attributed training point.

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of
    similarity-based explanations. In International Conference on Learning
    Representations.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        train_subclass_labels: torch.Tensor,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        filter_by_prediction: bool = False,
    ):
        """Initialize the Subclass Detection metric.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : torch.utils.data.Dataset
            The training dataset that was used to train `model`.
        train_subclass_labels : torch.Tensor
            The subclass labels of the training dataset.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        filter_by_prediction : bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the correct superclass is predicted, by
            default False.

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )

        assert len(train_subclass_labels) == ds_len(self.train_dataset), (
            f"Number of subclass labels ({len(train_subclass_labels)}) "
            f"does not match the number of train dataset samples "
            f"({ds_len(self.train_dataset)})."
        )
        self.subclass_labels = train_subclass_labels
        self.filter_by_prediction = filter_by_prediction

    def update(
        self,
        explanations: torch.Tensor,
        test_labels: Union[List[int], torch.Tensor],
        test_data: Optional[torch.Tensor] = None,
        grouped_labels: Optional[torch.Tensor] = None,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations of the test samples.
        test_labels : torch.Tensor
            Original labels of the test samples.
        test_data: Optional[torch.Tensor]
            Test samples to used to generate the explanations.
            Only required if `filter_by_prediction` is True during
            initalization.
        grouped_labels: Optional[torch.Tensor]
            The true superclasses of the test samples. Only required if
            `filter_by_prediction` is True during initalization.

        Raises
        ------
        ValueError
            If `test_data` and `grouped_labels` are not provided when
            `filter_by_prediction` is True.

        """
        explanations = explanations.to(self.device)

        if (
            test_data is None or grouped_labels is None
        ) and self.filter_by_prediction:
            raise ValueError(
                "test_data and grouped_labels must be provided if "
                "filter_by_prediction is True"
            )

        if isinstance(test_labels, list):
            test_labels = torch.tensor(test_labels)
        test_labels = test_labels.to(self.device)

        if test_data is not None:
            test_data = test_data.to(self.device)
        if grouped_labels is not None:
            if isinstance(grouped_labels, list):
                grouped_labels = torch.tensor(grouped_labels)
            grouped_labels = grouped_labels.to(self.device)

        select_idx = torch.tensor([True] * len(explanations)).to(self.device)
        if self.filter_by_prediction:
            pred_cls = self.model(test_data).argmax(dim=1)
            select_idx *= pred_cls == grouped_labels

        explanations = explanations[select_idx]
        test_labels = test_labels[select_idx].to(self.device)

        top_one_xpl_indices = explanations.argmax(dim=1)
        top_one_xpl_targets = torch.stack(
            [self.subclass_labels[int(i)] for i in top_one_xpl_indices]
        ).to(self.device)

        score = (test_labels == top_one_xpl_targets) * 1.0
        self.scores.append(score)
