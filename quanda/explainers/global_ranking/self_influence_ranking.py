"""Strategies for global ranking computation from a given local explainer."""

import warnings
from functools import lru_cache
from typing import Optional

import torch

from quanda.explainers.base import Explainer


class SelfInfluenceRanking:
    """Compute the self-influence-based ranking of training data."""

    def __init__(
        self,
        explainer: Optional[Explainer] = None,
        self_influence: Optional[torch.Tensor] = None,
    ):
        """Initialize the `SelfInfluenceRanking` class.

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
        if explainer is None and self_influence is None:
            raise ValueError(
                "An explainer of type BaseExplainer or a precomputed "
                "self_influence tensor is required for a metric with "
                "global method 'self-influence'."
            )
        self.explainer = explainer
        self._self_influence = self_influence

    def get_self_influence(self):
        """Compute self-influences using `self.explainer`.

        Returns
        -------
        torch.Tensor
            A 1D tensor containing the self-influence scores.

        """
        if self._self_influence is not None:
            return self._self_influence
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
        self_influence = self.get_self_influence()
        indices = torch.arange(
            self_influence.numel(),
            dtype=self_influence.dtype,
            device=self_influence.device,
        )
        self_influence += indices.to(self_influence.dtype) * 1e-4
        # TODO: this is done because sorting is not stable
        # TODO: find a better solution
        return torch.argsort(self_influence, stable=True)

    @staticmethod
    def _si_warning(method_name: str):
        """Print a warning message for unsupported methods."""
        warnings.warn(
            f"{method_name} method is not supported for "
            "`MislabelingDetectionMetric`. "
        )

    def reset(self, *args, **kwargs):
        """Reset the explainer state."""
        self._si_warning("reset")

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """Load the state dictionary."""
        self._si_warning("load_state_dict")

    def state_dict(self, *args, **kwargs):
        """Return the state dictionary."""
        self._si_warning("state_dict")
