from typing import List, Union

import torch
from captum.influence import DataInfluence

from src.explainers.base import Explainer


class CaptumExplainerWrapper(Explainer):
    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        train_dataset: torch.data.utils.Dataset,
        device: Union[str, torch.device],
        explainer_cls: DataInfluence,
        **explainer_kwargs,
    ):
        super().__init__(model=model, model_id=model_id, train_dataset=train_dataset, device=device)
        self.captum_explainer = explainer_cls(model=model, train_dataset=train_dataset, **explainer_kwargs)

    def explain(
        self, test: torch.Tensor, targets: Union[torch.Tensor, List[int], None], **explainer_kwargs
    ) -> torch.Tensor:
        if targets is not None:
            if not isinstance(targets, torch.Tensor):
                if isinstance(targets, list):
                    targets = torch.tensor(targets)
                else:
                    raise TypeError(
                        f"targets should be of type NoneType, List or torch.Tensor. Got {type(targets)} instead."
                    )
            return self.captum_explainer.influence(inputs=(test, targets), **explainer_kwargs)
        else:
            return self.captum_explainer.influence(inputs=test, **explainer_kwargs)
