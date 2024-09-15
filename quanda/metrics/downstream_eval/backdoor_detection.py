from typing import Dict, List, Optional

import torch

from quanda.metrics.base import Metric
from quanda.tasks.aggregate_attributions import AggregateAttributions


class BackdoorPoisoningDetectionMetric(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        poisoned_indices: List[int],
        poisoned_label: int,
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        model_id: Optional[str] = "0",
        cache_dir: str = "./cache",
        *args,
        **kwargs,
    ):
        super().__init__(model=model, train_dataset=train_dataset)
        self.scores: Dict[str, List[torch.Tensor]] = {k: [] for k in ["poisoned", "clean", "rest"]}
        clean_indices = [
            i for i in range(self.dataset_length) if (i not in poisoned_indices) and train_dataset[i][1] == poisoned_label
        ]
        rest_indices = list(set(range(self.dataset_length)) - set(poisoned_indices) - set(clean_indices))
        self.task = AggregateAttributions(
            model=model,
            train_dataset=train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            model_id=model_id,
            cache_dir=cache_dir,
            aggr_indices={"poisoned": poisoned_indices, "clean": clean_indices, "rest": rest_indices},
        )
        self.poisoned_indices = poisoned_indices

    def update(self, test_labels: torch.Tensor, explanations: torch.Tensor):
        """
        Used to implement metric-specific logic.
        """

        assert (
            test_labels.shape[0] == explanations.shape[0]
        ), f"Number of explanations ({explanations.shape[0]}) does not match the number of labels ({test_labels.shape[0]})."

        test_labels = test_labels.to(self.device)
        explanations = explanations.to(self.device)

        results = self.task.update(explanations=explanations, return_intermediate=True)

        for k, v in results.items():
            self.scores[k].append(v)

    def compute(self):
        """
        Used to aggregate current results and return a metric score.
        """
        return {k: torch.cat(self.scores[k]).mean().item() for k in self.scores.keys()}

    def reset(self, *args, **kwargs):
        """
        Used to reset the metric state.
        """
        {k: [] for k in ["poisoned", "clean", "rest"]}

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
