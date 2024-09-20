from typing import List, Optional

import torch

from quanda.metrics.base import Metric


class ClassDetectionMetric(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        model_id: Optional[str] = "0",
        cache_dir: str = "./cache",
        *args,
        **kwargs,
    ):
        super().__init__(model=model, train_dataset=train_dataset)
        self.scores: List[torch.Tensor] = []

    def update(self, test_labels: torch.Tensor, explanations: torch.Tensor):
        """
        Used to implement metric-specific logic.
        """

        assert (
            test_labels.shape[0] == explanations.shape[0]
        ), f"Number of explanations ({explanations.shape[0]}) does not match the number of labels ({test_labels.shape[0]})."

        test_labels = test_labels.to(self.device)
        explanations = explanations.to(self.device)

        _, top_one_xpl_indices = explanations.topk(k=1, dim=1)
        top_one_xpl_targets = torch.stack(
            [torch.tensor([self.train_dataset[i][1] for i in indices]).to(self.device) for indices in top_one_xpl_indices]
        ).squeeze()
        scores = (test_labels == top_one_xpl_targets) * 1.0
        self.scores.append(scores)

    def compute(self):
        """
        Used to aggregate current results and return a metric score.
        """
        return {"score": torch.cat(self.scores).mean().item()}

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
