from typing import List, Optional

import torch

from quanda.metrics.base import Metric
from quanda.tasks.proponents_per_sample import ProponentsPerSample


class ClassDetectionMetric(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        model_id: Optional[str] = "0",
        cache_dir: Optional[str] = "./cache",
        *args,
        **kwargs,
    ):
        super().__init__(model=model, train_dataset=train_dataset)
        self.scores: List[torch.Tensor] = []
        self.task = ProponentsPerSample(
            model=model,
            train_dataset=train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            model_id=model_id,
            cache_dir=cache_dir,
            top_k=1,
        )

    def update(self, test_labels: torch.Tensor, explanations: torch.Tensor):
        """
        Used to implement metric-specific logic.
        """

        assert (
            test_labels.shape[0] == explanations.shape[0]
        ), f"Number of explanations ({explanations.shape[0]}) exceeds the number of test labels ({test_labels.shape[0]})."

        test_labels = test_labels.to(self.device)
        explanations = explanations.to(self.device)

        top_one_xpl_indices = self.task.update(explanations=explanations, return_intermediate=True)
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
