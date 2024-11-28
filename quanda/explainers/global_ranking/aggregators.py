"""Module containing classes for creating global rankings."""

from abc import ABC, abstractmethod
from typing import Optional

import torch


class BaseAggregator(ABC):
    """Base class for attribution aggregators.

    Aggregators take local explanations and output a global ranking using
    different aggregation strategies.
    """

    def __init__(self):
        """Initialize the `BaseAggregator` base class."""
        self.scores: Optional[torch.Tensor] = None

    @abstractmethod
    def update(self, explanations: torch.Tensor):
        """Update the aggregator with new explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            The explanations to be aggregated.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        """
        raise NotImplementedError

    def _validate_explanations(self, explanations: torch.Tensor):
        """Validate the explanations tensor.

        Parameters
        ----------
        explanations : torch.Tensor
            The explanations tensor to be validated.

        Raises
        ------
        ValueError
            If the shape of explanations does not match the expected shape.

        """
        if self.scores is None:
            self.scores = torch.zeros(explanations.shape[1]).to(
                explanations.device
            )

        if explanations.shape[1] != self.scores.shape[0]:
            raise ValueError(
                f"Explanations shape {explanations.shape} does not match the "
                f"expected shape {self.scores.shape}"
            )

    def reset(self, *args, **kwargs):
        """Reset the aggregator state.

        Parameters
        ----------
        *args
            Variable length argument list.
        **kwargs
            Arbitrary keyword arguments.

        """
        self.scores = None

    def load_state_dict(self, state_dict: dict):
        """Load the aggregator state from a dictionary.

        Parameters
        ----------
        state_dict : dict
            The dictionary containing the state of the aggregator.

        """
        self.scores = state_dict["scores"]

    @property
    def state_dict(self, *args, **kwargs):
        """Return the aggregator state as a dictionary.

        Returns
        -------
        dict
            The dictionary containing the state of the aggregator.

        """
        return {"scores": self.scores}

    def compute(self) -> torch.Tensor:
        """Compute the aggregated scores.

        Returns
        -------
        torch.Tensor
            The aggregated scores.

        Raises
        ------
        ValueError
            If there are no scores to aggregate.

        """
        if self.scores is None:
            raise ValueError("No scores to aggregate.")
        return self.scores


class SumAggregator(BaseAggregator):
    """Aggregator which directly sums up the attributions."""

    def update(self, explanations: torch.Tensor):
        """Update the aggregated scores with the given explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            The explanations to be added to the aggregated scores.

        """
        self._validate_explanations(explanations)
        self.scores += explanations.sum(dim=0)


class AbsSumAggregator(BaseAggregator):
    """Aggregator which sums up the absolute value of attributions."""

    def update(self, explanations: torch.Tensor):
        """Update the aggregated scores with the given explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            The explanations to be added to the aggregated scores.

        """
        self._validate_explanations(explanations)
        self.scores += explanations.abs().sum(dim=0)


aggr_types = {
    "sum": SumAggregator,
    "sum_abs": AbsSumAggregator,
}
