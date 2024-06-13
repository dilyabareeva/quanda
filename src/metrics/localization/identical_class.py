import torch

from src.metrics.base import Metric


class IdenticalClass(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        device,
        *args,
        **kwargs,
    ):
        super().__init__(model, train_dataset, device, *args, **kwargs)
        self.scores = []

    def update(
        self,
        test_labels: torch.Tensor,
        explanations: torch.Tensor
    ):
        """
        Used to implement metric-specific logic.
        """

        assert (
            test_labels.shape[0] == explanations.shape[0]
        ), f"Number of explanations ({explanations.shape[0]}) exceeds the number of test labels ({test_labels.shape[0]})."

        top_one_xpl_indices = explanations.argmax(dim=1)
        top_one_xpl_targets = torch.stack([self.train_dataset[i][1] for i in top_one_xpl_indices])

        score = (test_labels == top_one_xpl_targets) * 1.0
        self.scores.append(score)

    def compute(self):
        """
        Used to aggregate current results and return a metric score.
        """
        return torch.cat(self.scores).mean()

    def reset(self, *args, **kwargs):
        """
        Used to reset the metric state.
        """
        self.scores = []

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Used to load the metric state.
        """
        self.scores = state_dict["scores"]

    def state_dict(self, *args, **kwargs):
        """
        Used to return the metric state.
        """
        return {"scores": self.scores}
