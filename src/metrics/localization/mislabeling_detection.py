from typing import List, Optional, Union

import torch

from src.explainers.aggregators import BaseAggregator
from src.explainers.base import BaseExplainer
from src.metrics.base import GlobalMetric
from src.utils.common import auc, cumsum


class MislabelingDetectionMetric(GlobalMetric):
    """
    Quote from https://proceedings.mlr.press/v89/khanna19a.html:

    'Our goal in this experiment is to try to identify some such misleading training data points,
    and remove them to see if it improves predictive accuracy. To illustrate the flexibility of
    our approach, we focus only on the digits 4 and 9 in the test data which were misclassified
    by our model, and then select the training data points responsible for those misclassifications.'

    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        poisoned_indices: List[int],
        global_method: Union[str, BaseAggregator] = "self-influence",
        explainer: Optional[BaseExplainer] = None,
        expl_kwargs: Optional[dict] = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            global_method=global_method,
            explainer=explainer,
            expl_kwargs=expl_kwargs,
            device=device,
        )
        self.poisoned_indices = poisoned_indices

    @classmethod
    def self_influence_based(
        cls,
        train_dataset: torch.utils.data.Dataset,
        poisoned_indices: List[int],
        explainer: BaseExplainer,
        expl_kwargs: Optional[dict] = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        return cls(
            model=explainer.model,
            poisoned_indices=poisoned_indices,
            train_dataset=train_dataset,
            global_method="self-influence",
            explainer=explainer,
            expl_kwargs=expl_kwargs,
            device=device,
        )

    @classmethod
    def aggr_based(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        poisoned_indices: List[int],
        aggregator: Union[str, BaseAggregator],
        trainer_fit_kwargs: Optional[dict] = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        return cls(
            model=model,
            global_method=aggregator,
            poisoned_indices=poisoned_indices,
            train_dataset=train_dataset,
            device=device,
        )

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        self.strategy.update(explanations, **kwargs)

    def reset(self, *args, **kwargs):
        self.strategy.reset()

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.strategy.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        return self.strategy.state_dict()

    def compute(self, *args, **kwargs):

        global_ranking = self.strategy.get_global_rank()
        success_arr = torch.tensor([elem in self.poisoned_indices for elem in global_ranking])
        unnormalized_curve = cumsum(success_arr * 1.0)
        return {
            "success_arr": success_arr,
            "score": auc(cumsum(success_arr * 1.0), max=len(self.poisoned_indices)),
            "curve": unnormalized_curve / len(self.poisoned_indices),
        }
