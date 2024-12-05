"""Strategies for global ranking computation."""

import warnings
from functools import lru_cache
from typing import Optional

import torch

from quanda.explainers.base import Explainer
from quanda.explainers.global_ranking.aggregators import BaseAggregator


class GlobalSelfInfluenceStrategy:
    """Compute the self-influence-based ranking of training data."""

    def __init__(
        self,
        explainer: Optional[Explainer] = None,
    ):
        """Initialize the `GlobalSelfInfluenceStrategy` class.

        This class is used to generate a ranking of training data ordered by
        their self-influence scores,
        using an `Explainer` object.

        Parameters
        ----------
        explainer : Optional[Explainer], optional
            `Explainer` object to use for self-influence computation, by
            default None

        Raises
        ------
        ValueError
            If `explainer` is not provided.

        """
        if explainer is None:
            raise ValueError(
                "An explainer of type BaseExplainer is required for a metric "
                "with global method 'self-influence'."
            )
        self.explainer = explainer

    def get_self_influence(self):
        """Compute self-influences using `self.explainer`.

        Returns
        -------
        torch.Tensor
            A 1D tensor containing the self-influence scores.

        """
        return self.explainer.self_influence()

    @lru_cache(maxsize=1)
    def get_global_rank(self):
        """Return the global ranking of training data.

        Ordered by the self-influence scores. Cached to avoid recomputation.

        Returns
        -------
        torch.Tensor
            A 1D tensor containing the global ranking.

        """
        return self.get_self_influence().argsort()

    @staticmethod
    def _si_warning(method_name: str):
        """Print a warning message for unsupported methods."""
        warnings.warn(
            f"{method_name} method is not supported for a metric with global "
            f"method 'self-influence'. Method call will be ignored. Call "
            f"'compute' method to get the final result."
        )

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        """Update the explainer with the given explanations."""
        self._si_warning("update")

    def reset(self, *args, **kwargs):
        """Reset the explainer state."""
        pass

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """Load the state dictionary."""
        self._si_warning("load_state_dict")

    def state_dict(self, *args, **kwargs):
        """Return the state dictionary."""
        self._si_warning("state_dict")


class GlobalAggrStrategy:
    """Compute the global ranking of training data, by aggregating local."""

    def __init__(
        self,
        aggr_type: type,
    ):
        """Initialize the `GlobalAggrStrategy` class.

        Parameters
        ----------
        aggr_type : type
            The type of the aggregator to be used for global ranking
            computation.

        Raises
        ------
        ValueError
            If the aggregator type is not a subclass of `BaseAggregator`.

        """
        self.aggregator = aggr_type()
        self.global_rank: torch.Tensor

        if not isinstance(self.aggregator, BaseAggregator):
            raise ValueError(f"Aggregator type {aggr_type} is not supported.")

    def update(
        self,
        explanations: torch.Tensor,
    ):
        """Update the aggregator with the given explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            The local attributions to be aggregated.

        """
        self.aggregator.update(explanations)

    def get_global_rank(self, *args, **kwargs):
        """Compute global rank according to the aggregated scores.

        Returns
        -------
        torch.Tensor
            A 1D tensor containing the global ranking.

        """
        return self.aggregator.compute().argsort()

    def reset(self, *args, **kwargs):
        """Reset the aggregator state."""
        self.aggregator.reset()

    def load_state_dict(self, state_dict: dict):
        """Load the aggregator state from a dictionary.

        Parameters
        ----------
        state_dict : dict
            State dictionary to load.

        """
        self.aggregator.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        """Returnthe aggregator state as a dictionary.

        Returns
        -------
        dict
            The state dictionary.

        """
        return self.aggregator.state_dict()
