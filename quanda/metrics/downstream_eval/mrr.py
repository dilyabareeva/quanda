"""Mean Reciprocal Rank Metric."""

from typing import Any, Callable, List, Optional, Union

import datasets  # type: ignore
import torch

from quanda.metrics.base import Metric


class MRRMetric(Metric):
    """Mean Reciprocal Rank (MRR) Metric.

    This metric measures the mean reciprocal rank of the highest-ranked
    entailing proponent for each fact query.

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
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
    ):
        """Initialize the MRR metric.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : Union[torch.utils.data.Dataset, datasets.Dataset]
            The training dataset that was used to train `model`.
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

        self.reciprocal_ranks: List[torch.Tensor] = []

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

        # Compute MRR for each query
        ranks = torch.zeros(num_queries, device=self.device)
        for i in range(num_queries):
            # Find rank of first entailing proponent
            rank_indices = torch.nonzero(
                entailment_labels[i][sorted_indices[i]], as_tuple=True
            )[0]
            if len(rank_indices) > 0:
                first_rank = (
                    rank_indices[0].item() + 1
                )  # +1 because rank is 1-indexed
                ranks[i] = 1.0 / first_rank

        self.reciprocal_ranks.append(ranks)

    def compute(self, *args, **kwargs):
        """Compute the MRR metric.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the metric score.

        """
        return {"score": torch.cat(self.reciprocal_ranks).mean().item()}

    def reset(self, *args, **kwargs):
        """Reset the metric state."""
        self.reciprocal_ranks = []

    def load_state_dict(self, state_dict: dict):
        """Load previously computed state for the metric.

        Parameters
        ----------
        state_dict : dict
            A state dictionary for the metric.

        """
        self.reciprocal_ranks = state_dict["reciprocal_ranks"]

    def state_dict(self, *args, **kwargs):
        """Return the metric state.

        Returns
        -------
        dict
            The state dictionary containing the reciprocal ranks.

        """
        return {
            "reciprocal_ranks": self.reciprocal_ranks,
        }
