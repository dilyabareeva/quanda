"""Top-K Cardinality Metric."""

from typing import Optional, Union, List, Callable, Any

import torch

from quanda.metrics.base import Metric


class TopKCardinalityMetric(Metric):
    """Top-K Cardinality Metric.

    Metric that measures the overlap of top-k explanations across all test
    samples, as argued in Barshan et al. (2020). A good data attributor's
    attributions should depend on the input data, and thus should have low
    overlap between test points.

    References
    ----------
    1) Barshan, Elnaz, Marc-Etienne Brunet, and Gintare Karolina Dziugaite.
    (2020). Relatif: Identifying explanatory training samples via relative
    influence. International Conference on Artificial Intelligence and
    Statistics. PMLR.

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        top_k: int = 1,
    ):
        """Initialize the Top-K Cardinality metric.

        Parameters
        ----------
        model : torch.nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : torch.utils.data.Dataset
            The training dataset that was used to train `model`.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        top_k : int, optional
            The number of top-k explanations to consider, defaults to 1.

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )

        self.top_k = top_k
        self.all_top_k_examples = torch.empty(0, top_k).to(self.device)

    def update(
        self,
        explanations: torch.Tensor,
    ):
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations of the test samples.

        """
        explanations = explanations.to(self.device)

        _, top_k_indices = explanations.topk(k=self.top_k, dim=1)
        self.all_top_k_examples = torch.concat(
            (self.all_top_k_examples, top_k_indices), dim=0
        )

    def compute(self, *args, **kwargs):
        """Compute the metric score.

        Returns
        -------
        dict
            A dictionary containing the metric score in the `score` field.

        """
        return {
            "score": len(torch.unique(self.all_top_k_examples))
            / torch.numel(self.all_top_k_examples)
        }

    def reset(self, *args, **kwargs):
        """Reset the metric state."""
        self.all_top_k_examples = torch.empty(0, self.top_k)

    def load_state_dict(self, state_dict: dict):
        """Load the metric state from a state dictionary.

        Parameters
        ----------
        state_dict : dict
            A state dictionary for the metric.

        """
        self.all_top_k_examples = state_dict["all_top_k_examples"]

    def state_dict(self, *args, **kwargs):
        """Return the metric state as a dictionary.

        Returns
        -------
        dict
            A dictionary containing the metric state.

        """
        return {"all_top_k_examples": self.all_top_k_examples}
