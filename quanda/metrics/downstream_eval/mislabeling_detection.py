from typing import Any, List, Optional, Union

import torch

from quanda.metrics.base import Metric
from quanda.tasks.global_ranking import GlobalRanking


class MislabelingDetectionMetric(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        poisoned_indices: List[int],
        global_method: Union[str, type] = "self-influence",
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
        )
        self.global_ranker = GlobalRanking(
            model=model,
            train_dataset=train_dataset,
            global_method=global_method,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            model_id="test",
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
        *args: Any,
        **kwargs: Any,
    ):
        return cls(
            model=model,
            poisoned_indices=poisoned_indices,
            train_dataset=train_dataset,
            global_method="self-influence",
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

    @classmethod
    def aggr_based(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        poisoned_indices: List[int],
        aggregator_cls: Union[str, type],
        *args,
        **kwargs,
    ):
        return cls(
            model=model,
            global_method=aggregator_cls,
            poisoned_indices=poisoned_indices,
            train_dataset=train_dataset,
        )

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        self.global_ranker.update(explanations, **kwargs)

    def reset(self, *args, **kwargs):
        self.global_ranker.reset()

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.global_ranker.load_state_dict(state_dict)

    def state_dict(self, *args, **kwargs):
        return self.global_ranker.state_dict()

    def compute(self, *args, **kwargs):
        global_ranking = self.global_ranker.compute()
        success_arr = torch.tensor([elem in self.poisoned_indices for elem in global_ranking])
        normalized_curve = torch.cumsum(success_arr * 1.0, dim=0) / len(self.poisoned_indices)
        score = torch.trapezoid(normalized_curve) / len(self.poisoned_indices)
        return {
            "score": score.item(),
            "success_arr": success_arr,
            "curve": normalized_curve / len(self.poisoned_indices),
        }