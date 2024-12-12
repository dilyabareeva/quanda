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
        test_subclasses: Union[List[int], torch.Tensor],
        test_tensor: Optional[torch.Tensor] = None,
        test_classes: Optional[torch.Tensor] = None,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations of the test samples.
        test_subclasses : torch.Tensor
            Original labels of the test samples.
        test_tensor: Optional[torch.Tensor]
            Test samples to used to generate the explanations.
            Only required if `filter_by_prediction` is True during
            initalization.
        test_classes: Optional[torch.Tensor]
            The true superclasses of the test samples. Only required if
            `filter_by_prediction` is True during initalization.

        Raises
        ------
        ValueError
            If `test_tensor` and `test_classes` are not provided when
            `filter_by_prediction` is True.

        """
        explanations = explanations.to(self.device)

        if (
            test_tensor is None or test_classes is None
        ) and self.filter_by_prediction:
            raise ValueError(
                "test_tensor and test_classes must be provided if "
                "filter_by_prediction is True"
            )

        if isinstance(test_subclasses, list):
            test_subclasses = torch.tensor(test_subclasses)
        test_subclasses = test_subclasses.to(self.device)

        if test_tensor is not None:
            test_tensor = test_tensor.to(self.device)
        if test_classes is not None:
            if isinstance(test_classes, list):
                test_classes = torch.tensor(test_classes)
            test_classes = test_classes.to(self.device)

        select_idx = torch.tensor([True] * len(explanations)).to(self.device)
        if self.filter_by_prediction:
            pred_cls = self.model(test_tensor).argmax(dim=1)
            select_idx *= pred_cls == test_classes

        explanations = explanations[select_idx]
        test_subclasses = test_subclasses[select_idx].to(self.device)

        top_one_xpl_indices = explanations.argmax(dim=1)
        top_one_xpl_targets = torch.stack(
            [self.subclass_labels[int(i)] for i in top_one_xpl_indices]
        ).to(self.device)

        score = (test_subclasses == top_one_xpl_targets) * 1.0
        self.scores.append(score)
