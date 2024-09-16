from typing import Dict, List, Optional

import torch

from quanda.metrics.base import Metric
from quanda.tasks.aggregate_attributions import AggregateAttributions


class ShortcutDetectionMetric(Metric):
    """Metric for the shortcut detection evaluation task.
    Attributions of a  model with a shortcut is checked against the ground truth of shortcut samples.
    This strategy is inspired by (1) and (2).

    References
    ----------
    1) Koh, Pang Wei, and Percy Liang. "Understanding black-box predictions via influence functions."
        International conference on machine learning. PMLR, 2017.
    2) Søgaard, Anders. "Revisiting methods for finding influential examples." arXiv preprint arXiv:2111.04683 (2021).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        poisoned_indices: List[int],
        poisoned_cls: int,
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        model_id: Optional[str] = "0",
        cache_dir: str = "./cache",
        *args,
        **kwargs,
    ):
        """Initializer for the Shortcut Detection metric.

        Parameters
        ----------
        model : torch.nn.Module
            Model associated with the attributions to be evaluated.
        train_dataset : torch.utils.data.Dataset
            Training dataset used to train `model`.
        poisoned_indices : List[int]
            Ground truth of shortcut indices of the `train_dataset`.
        poisoned_cls : int
            Class of the poisoned samples.
        explainer_cls : Optional[type], optional
            Optional explainer class to be evaluated, defaults to None
        expl_kwargs : Optional[dict], optional
            Optional keyword arguments for explainers, defaults to None
        model_id : Optional[str], optional
            Optional model_id
        cache_dir : str, optional
            Optional cache directory
        """
        super().__init__(model=model, train_dataset=train_dataset)
        self.scores: Dict[str, List[torch.Tensor]] = {k: [] for k in ["poisoned", "clean", "rest"]}
        clean_indices = [
            i for i in range(self.dataset_length) if (i not in poisoned_indices) and train_dataset[i][1] == poisoned_cls
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

    def update(self, explanations: torch.Tensor):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations to be evaluated.
        """
        explanations = explanations.to(self.device)

        results = self.task.update(explanations=explanations, return_intermediate=True)

        for k, v in results.items():
            self.scores[k].append(v)

    def compute(self):
        """
        Aggregates current results and return a metric score.
        """
        additional_results = {k: torch.cat(self.scores[k]).mean().item() for k in ["clean", "rest"]}
        return {"score": torch.cat(self.scores["poisoned"]).mean().item(), **additional_results}

    def reset(self, *args, **kwargs):
        """
        Resets the metric state.
        """
        {k: [] for k in ["poisoned", "clean", "rest"]}

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Loads the metric state.
        """
        self.scores = state_dict["scores"]

    def state_dict(self, *args, **kwargs):
        """
        Returns the metric state.
        """
        return {"scores": self.scores}