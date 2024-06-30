from typing import List

import torch

from src.metrics.base import Metric


class BackdoorDetectionMetric(Metric):
    """ """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        backdoor_indices: List[int],
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            device=device,
        )
        self.backdoor_indices = backdoor_indices
        self.scores: List[torch.Tensor] = []

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        explanations = explanations.to(self.device)

        top_one_xpl_indices = explanations.argmax(dim=1)

        scores = (torch.tensor([i in self.backdoor_indices for i in top_one_xpl_indices])) * 1.0
        self.scores.append(scores)

    def reset(self, *args, **kwargs):
        self.scores = []

    def compute(self, *args, **kwargs):
        return torch.cat(self.scores).mean()

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


0.1250, 0.2500, 0.3750, 0.5000, 0.6250, 0.7500, 0.8750, 1.0000
