from typing import Union

import torch
from captum.influence import SimilarityInfluence

from src.explainers.captum.base import CaptumExplainerWrapper


class CaptumSimilarityExplainer(CaptumExplainerWrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: str,
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        layer: str,
        **explainer_init_kwargs,
    ):
        self.layer = layer
        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
            explainer_cls=SimilarityInfluence,
            layers=[layer],
            **explainer_init_kwargs,
        )

    def initialize_captum(self, cls, **init_kwargs):
        self.captum_explainer = cls(
            module=self.model,
            influence_src_dataset=self.train_dataset,
            activation_dir=self.cache_dir,
            model_id=self.model_id,
            similarity_direction="max",
            **init_kwargs,
        )

    def load_state_dict(self, path):
        return

    def reset(self):
        return

    def explain(self, test: torch.Tensor) -> torch.Tensor:
        topk_idx, topk_val = super().explain(test=test, targets=None, top_k=len(self.train_dataset))[self.layer]
        inverted_idx = topk_idx.argsort()
        tda = torch.cat([topk_val[None, i, inverted_idx[i]] for i in range(topk_idx.shape[0])], dim=0)
        return tda
