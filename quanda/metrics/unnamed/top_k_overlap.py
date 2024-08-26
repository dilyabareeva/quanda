from typing import Optional

import torch

from quanda.metrics.base import Metric
from quanda.tasks.proponents_per_sample import ProponentsPerSample


class TopKOverlapMetric(Metric):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
            explainer_cls: Optional[type] = None,
            expl_kwargs: Optional[dict] = None,
            model_id: Optional[str] = "0",
            cache_dir: Optional[str] = "./cache",
        top_k: int = 1,
        *args,
        **kwargs,
    ):
        super().__init__(model=model, train_dataset=train_dataset)
        self.task = ProponentsPerSample(
            model=model,
            train_dataset=train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            model_id=model_id,
            cache_dir=cache_dir,
            top_k=top_k,
        )
        self.top_k = top_k
        self.all_top_k_examples = torch.empty(0, top_k).to(self.device)

    def update(
        self,
        explanations: torch.Tensor,
        **kwargs,
    ):
        explanations = explanations.to(self.device)

        top_k_indices = self.task.update(explanations=explanations, return_intermediate=True)
        self.all_top_k_examples = torch.concat((self.all_top_k_examples, top_k_indices), dim=0)

    def compute(self, *args, **kwargs):
        return {"score": len(torch.unique(self.all_top_k_examples))}

    def reset(self, *args, **kwargs):
        self.all_top_k_examples = torch.empty(0, self.top_k)

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.all_top_k_examples = state_dict["all_top_k_examples"]

    def state_dict(self, *args, **kwargs):
        return {"all_top_k_examples": self.all_top_k_examples}
