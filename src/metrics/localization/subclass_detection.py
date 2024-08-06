import torch

from src.metrics.localization import ClassDetectionMetric


class SubclassDetectionMetric(ClassDetectionMetric):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        subclass_labels: torch.Tensor,
        device,
        *args,
        **kwargs,
    ):
        super().__init__(model, train_dataset, device, *args, **kwargs)
        assert len(subclass_labels) == self.dataset_length, (
            f"Number of subclass labels ({len(subclass_labels)}) "
            f"does not match the number of train dataset samples ({self.dataset_length})."
        )
        self.subclass_labels = subclass_labels

    def update(self, test_subclasses: torch.Tensor, explanations: torch.Tensor):
        """
        Used to implement metric-specific logic.
        """

        assert (
            test_subclasses.shape[0] == explanations.shape[0]
        ), f"Number of explanations ({explanations.shape[0]}) exceeds the number of test labels ({test_subclasses.shape[0]})."

        top_one_xpl_indices = explanations.argmax(dim=1)
        top_one_xpl_targets = torch.stack([self.subclass_labels[i] for i in top_one_xpl_indices])

        score = (test_subclasses == top_one_xpl_targets) * 1.0
        self.scores.append(score)
