from typing import Any, Dict, List, Optional

import torch

from quanda.tasks.base import Task


class AggregateAttributions(Task):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        aggr_indices: Dict[str, List[int]],
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        model_id: Optional[str] = "0",
        cache_dir: str = "./cache",
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            model_id=model_id,
            cache_dir=cache_dir,
        )
        self.aggr_indices = aggr_indices
        self.result: Dict[str, List[torch.Tensor]] = {k: [] for k in self.aggr_indices.keys()}

    def update(
        self,
        explanations: torch.Tensor,
        return_intermediate: bool = False,
        *args: Any,
        **kwargs: Any,
    ):

        explanations = explanations.to(self.device)
        immediate_return_dict = {}
        for k, ind in self.aggr_indices.items():
            if len(ind) > 0:
                aggr_attr = explanations[:, ind].mean(dim=1)
            else:
                aggr_attr = torch.zeros(explanations.shape[0], device=self.device)
            immediate_return_dict[k] = aggr_attr
            self.result[k].append(aggr_attr)

        if return_intermediate:
            return immediate_return_dict

    def explain_update(
        self,
        test_data: torch.Tensor,
        explanation_targets: torch.Tensor,
        explanations: torch.Tensor,
        return_intermediate: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Used to implement task-specific logic.
        """
        explanations = self.explainer.explain(
            test=test_data,
            targets=explanation_targets,
        )
        return self.update(explanations=explanations, return_intermediate=return_intermediate)

    def compute(self):
        """
        Used to aggregate current results and return a task result.
        """
        return {key: torch.cat(val) for key, val in self.result.items()}

    def reset(self, *args, **kwargs):
        """
        Used to reset the task state.
        """
        self.result = {k: [] for k in self.aggr_indices.keys()}

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Used to load the task state.
        """
        self.result = state_dict["results"]

    def state_dict(self, *args, **kwargs):
        """
        Used to return the task state.
        """
        return {"results": self.result}
