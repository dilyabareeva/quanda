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
        subclass_labels: torch.Tensor,
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
        assert len(subclass_labels) == self.dataset_length, (
            f"Number of subclass labels ({len(subclass_labels)}) "
            f"does not match the number of train dataset samples ({self.dataset_length})."
        )
        self.subclass_labels = subclass_labels

    def update(self, test_subclasses: torch.Tensor, explanations: torch.Tensor):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        test_subclasses : torch.Tensor
            Original labels of the test samples
        explanations : torch.Tensor
            Explanations of the test samples
        """

        assert (
            test_subclasses.shape[0] == explanations.shape[0]
        ), f"Number of explanations ({explanations.shape[0]}) exceeds the number of test labels ({test_subclasses.shape[0]})."

        top_one_xpl_indices = explanations.argmax(dim=1)
        top_one_xpl_targets = torch.stack([self.subclass_labels[i] for i in top_one_xpl_indices])

        score = (test_subclasses == top_one_xpl_targets) * 1.0
        self.scores.append(score)
