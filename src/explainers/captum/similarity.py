from typing import Union

import torch
from captum.influence import SimilarityInfluence

from src.explainers.captum.base import CaptumExplainerWrapper


class CaptumSimilarityExplainer(CaptumExplainerWrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        train_dataset: torch.data.utils.Dataset,
        device: Union[str, torch.device],
        **explainer_kwargs,
    ):
        super().__init__(
            model=model,
            model_id=model_id,
            train_dataset=train_dataset,
            device=device,
            explainer_cls=SimilarityInfluence,
            **explainer_kwargs,
        )

    def explain(self, test: torch.Tensor) -> torch.Tensor:
        return super().explain(test=test, targets=None)
