from typing import Any, List, Optional

import torch

from quanda.tasks.base import Task


class ProponentsPerSample(Task):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        top_k: int = 1,
        model_id: Optional[str] = "0",
        cache_dir: Optional[str] = "./cache",
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            model_id=model_id,
            cache_dir=cache_dir,
        )
        self.top_k = top_k
        self.result: List[torch.Tensor] = []

    def update(
        self,
        explanations: torch.Tensor,
        return_intermediate: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Used to implement task-specific logic.
        """

        explanations = explanations.to(self.device)

        top_k_values, top_k_xpl_indices = explanations.topk(k=self.top_k, dim=1)

        self.result.append(top_k_xpl_indices)
        if return_intermediate:
            return top_k_xpl_indices

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
        Used to aggregate current results and return a task score.
        """
        return {"score": torch.cat(self.result).mean().item()}

    def reset(self, *args, **kwargs):
        """
        Used to reset the task state.
        """
        self.result = []

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Used to load the task state.
        """
        self.result = state_dict["scores"]

    def state_dict(self, *args, **kwargs):
        """
        Used to return the task state.
        """
        return {"scores": self.result}
