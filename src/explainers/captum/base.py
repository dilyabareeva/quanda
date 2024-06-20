from typing import List, Optional, Union

import torch
from captum.influence import DataInfluence

from src.explainers.base import Explainer


class CaptumExplainerWrapper(Explainer):
    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        explainer_cls: DataInfluence,
        **explainer_init_kwargs,
    ):
        super().__init__(
            model=model, model_id=model_id, train_dataset=train_dataset, device=device, cache_dir=cache_dir
        )
        for shared_field_name in ["model_id", "cache_dir"]:
            assert shared_field_name not in explainer_init_kwargs.keys(), (
                f"{shared_field_name} is already given to the explainer object, "
                "it must not be repeated in the explainer_init_kwargs"
            )

        self.initialize_captum(explainer_cls, **explainer_init_kwargs)

    def initialize_captum(self, cls, **init_kwargs):
        self.captum_explainer = cls(model=self.model, train_dataset=self.train_dataset, **init_kwargs)

    def explain(
        self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]], **explain_fn_kwargs
    ) -> torch.Tensor:
        test = test.to(self.device)
        if targets is not None:
            if not isinstance(targets, torch.Tensor):
                if isinstance(targets, list):
                    targets = torch.tensor(targets)
                else:
                    raise TypeError(
                        f"targets should be of type NoneType, List or torch.Tensor. Got {type(targets)} instead."
                    )
            targets = targets.to(self.device)
            return self.captum_explainer.influence(inputs=(test, targets), **explain_fn_kwargs)
        else:
            return self.captum_explainer.influence(inputs=test, **explain_fn_kwargs)
