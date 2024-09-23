from typing import List, Optional, Union

import torch

from quanda.metrics.downstream_eval import ClassDetectionMetric


class SubclassDetectionMetric(ClassDetectionMetric):
    """Subclass Detection Metric as defined in (1).
    A model is trained on a dataset where labels are grouped into superclasses.
    The metric evaluates the performance of an attribution method in detecting the subclass of a test sample
    from its highest attributed training point.

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of similarity-based explanations. In International
    Conference on Learning Representations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        train_subclass_labels: torch.Tensor,
        filter_by_prediction: bool = False,
        *args,
        **kwargs,
    ):
        """Initializer for the Subclass Detection metric.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : torch.utils.data.Dataset
            The training dataset that was used to train `model`.
        subclass_labels : torch.Tensor
            The subclass labels of the training dataset.
        """
        super().__init__(model, train_dataset)
        assert len(train_subclass_labels) == self.dataset_length, (
            f"Number of subclass labels ({len(train_subclass_labels)}) "
            f"does not match the number of train dataset samples ({self.dataset_length})."
        )
        self.subclass_labels = train_subclass_labels
        self.filter_by_prediction = filter_by_prediction

    def update(
        self,
        test_subclasses: Union[List[int], torch.Tensor],
        explanations: torch.Tensor,
        test_tensor: Optional[torch.Tensor] = None,
        test_classes: Optional[torch.Tensor] = None,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        test_subclasses : torch.Tensor
            Original labels of the test samples
        explanations : torch.Tensor
            Explanations of the test samples
        """

        if (test_tensor is None or test_classes is None) and self.filter_by_prediction:
            raise ValueError("test_tensor and test_classes must be provided if filter_by_prediction is True")

        if isinstance(test_subclasses, list):
            test_subclasses = torch.tensor(test_subclasses)

        if test_tensor is not None:
            test_tensor = test_tensor.to(self.device)
        if test_classes is not None:
            if isinstance(test_classes, list):
                test_classes = torch.tensor(test_classes)
            test_classes = test_classes.to(self.device)

        select_idx = torch.tensor([True] * len(explanations))
        if self.filter_by_prediction:
            pred_cls = self.model(test_tensor).argmax(dim=1)
            select_idx *= pred_cls == test_classes

        explanations = explanations[select_idx].to(self.device)
        test_subclasses = test_subclasses[select_idx].to(self.device)

        top_one_xpl_indices = explanations.argmax(dim=1)
        top_one_xpl_targets = torch.stack([self.subclass_labels[int(i)] for i in top_one_xpl_indices]).to(self.device)

        score = (test_subclasses == top_one_xpl_targets) * 1.0
        self.scores.append(score)
