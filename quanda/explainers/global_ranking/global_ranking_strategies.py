import warnings
from functools import lru_cache
from typing import Optional

import torch

from quanda.explainers.base import Explainer
from quanda.explainers.global_ranking.aggregators import BaseAggregator


class GlobalSelfInfluenceStrategy:
    """
    This object is used in some metrics to compute the self-influence-based ranking of training data,
    using an `Explainer` object.
    """

    def __init__(
        self,
        explainer: Optional[Explainer] = None,
    ):
        """
        Initializer for the `GlobalSelfInfluenceStrategy` class.
        This class is used to generate a ranking of training data ordered by their self-influence scores,
        using an `Explainer` object.

        Parameters
        ----------
        explainer : Optional[Explainer], optional
            `Explainer` object to use for self-influence computation, by default None

        Raises
        ------
        ValueError
            If `explainer` is not provided.
        """
        if explainer is None:
            raise ValueError(
                "An explainer of type BaseExplainer is required for a metric with global method 'self-influence'."
            )
        self.explainer = explainer

    def get_self_influence(self):
        """
        Computes self-influences using `self.explainer`

        Returns
        -------
        torch.Tensor
            A 1D tensor containing the self-influence scores.
        """
        return self.explainer.self_influence()

    @lru_cache(maxsize=1)
    def get_global_rank(self):
        """
        Returns the global ranking of training data ordered by their self-influence scores.
        Cached to avoid recomputation.

        Returns
        -------
        torch.Tensor
            A 1D tensor containing the global ranking.
        """
        return self.get_self_influence().argsort()

    @staticmethod
    def _si_warning(method_name: str):
        warnings.warn(
            f"{method_name} method is not supported for a metric with global method "
            "'self-influence'. Method call will be ignored. Call 'compute' method to get the final result."
        )

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        self._si_warning("update")

    def reset(self, *args, **kwargs):
        pass

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self._si_warning("load_state_dict")

    def state_dict(self, *args, **kwargs):
        self._si_warning("state_dict")


class GlobalAggrStrategy:
    """
    This class is used in some metrics to compute the global ranking of training data,
    by aggregating supplied local attributions.
    """

    def __init__(
        self,
        aggr_type: type,
    ):
        """
        Initializer for the `GlobalAggrStrategy` class.

        Parameters
        ----------
        aggr_type : type
            The type of the aggregator to be used for global ranking computation.

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
        **kwargs,
    ):
        """
        Updates the aggregator with the given explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            The local attributions to be aggregated.
        """
        self.aggregator.update(explanations)

    def get_global_rank(self, *args, **kwargs):
        """
        Compute global rank according to the aggregated scores.

        Returns
        -------
        torch.Tensor
            A 1D tensor containing the global ranking.
        """
        return self.aggregator.compute().argsort()

    def reset(self, *args, **kwargs):
        """
        Reset the aggregator state.
        """
        self.aggregator.reset()

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Load the aggregator state from a dictionary

        Parameters
        ----------
        state_dict : dict
            State dictionary to load.
        """
        self.aggregator.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        """
        Returns the aggregator state as a dictionary.

        Returns
        -------
        dict
            The state dictionary.
        """
        return self.aggregator.state_dict()
