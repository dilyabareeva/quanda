from typing import List, Optional, Union

import torch

from src.explainers.aggregators import BaseAggregator
from src.explainers.base import BaseExplainer
from src.metrics.base import GlobalMetric
from src.utils.common import auc


class MislabelingDetectionMetric(GlobalMetric):

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        poisoned_indices: List[int],
        global_method: Union[str, BaseAggregator] = "self-influence",
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            global_method=global_method,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            device=device,
        )
        self.poisoned_indices = poisoned_indices

    @classmethod
    def self_influence_based(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        poisoned_indices: List[int],
        expl_kwargs: Optional[dict] = None,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        return cls(
            model=model,
            poisoned_indices=poisoned_indices,
            train_dataset=train_dataset,
            global_method="self-influence",
            explainer_cls=explainer_cls,
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
        unnormalized_curve = torch.cumsum(success_arr * 1.0, dim=0)
        return {
            "success_arr": success_arr,
            "score": auc(torch.cumsum(success_arr * 1.0, dim=0), max=len(self.poisoned_indices)),
            "curve": unnormalized_curve / len(self.poisoned_indices),
        }
