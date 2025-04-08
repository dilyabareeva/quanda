"""Recall@k Metric."""

from typing import Any, Callable, List, Optional, Union

import datasets  # type: ignore
import torch

from quanda.metrics.base import Metric


class RecallAtKMetric(Metric):
    """Recall@k Metric.

    This metric measures the proportion of facts for which an entailing
    proponent appears in the top k proponent retrievals.

    References
    ----------
    1) Tyler A. Chang, Dheeraj Rajagopal, Tolga Bolukbasi, Lucas Dixon,
    and Ian Tenney. (2024) "Scalable Influence and Fact Tracing for
    Large Language Model Pretraining". The Thirteenth International
    Conference on Learning Representations.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        k: int = 10,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
    ):
        """Initialize the Recall@k metric.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : Union[torch.utils.data.Dataset, datasets.Dataset]
            The training dataset that was used to train `model`.
        k : int, optional
            The k value for Recall@k, by default 10.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )

        self.k = k
        self.recalls: List[torch.Tensor] = []

    def update(
        self,
        explanations: torch.Tensor,
        entailment_labels: torch.Tensor,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Attributions for training examples for each query.
        entailment_labels : torch.Tensor
            Binary tensor indicating whether each proponent entails each query.

        """
        explanations = explanations.to(self.device)
        entailment_labels = entailment_labels.to(self.device)
        num_queries = explanations.shape[0]

        # Get the indices of attributions in descending order
        _, sorted_indices = torch.sort(explanations, dim=1, descending=True)

        # Compute Recall@k for each query
        recalls = torch.zeros(num_queries, device=self.device)
        for i in range(num_queries):
            # Check if any entailing proponent in top k
            top_k_indices = sorted_indices[i][: self.k]
            has_entailing_in_top_k = torch.any(
                entailment_labels[i][top_k_indices]
            ).item()
            recalls[i] = float(has_entailing_in_top_k)

        self.recalls.append(recalls)

    def compute(self, *args, **kwargs):
        """Compute the Recall@k metric.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the metric score.

        """
        if not self.recalls:
            return {"score": 0.0}

        return {"score": torch.cat(self.recalls).mean().item()}

    def reset(self, *args, **kwargs):
        """Reset the metric state."""
        self.recalls = []

    def load_state_dict(self, state_dict: dict):
        """Load previously computed state for the metric.

        Parameters
        ----------
        state_dict : dict
            A state dictionary for the metric.

        """
        self.recalls = state_dict["recalls"]

    def state_dict(self, *args, **kwargs):
        """Return the metric state."""
        return {
            "recalls": self.recalls,
        }
