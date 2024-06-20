from typing import List, Optional, Union, Any, Dict

import torch
from captum.influence import SimilarityInfluence

from src.explainers.base_explainer import BaseExplainer
from src.utils.validation import validate_1d_tensor_or_int_list


class CaptumInfluence(BaseExplainer):
    """
    TODO: should this inherit from BaseExplainer?
    Or should it just follow the same protocol?
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        explainer_cls: type,
        explain_kwargs: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(
            model=model, model_id=model_id, cache_dir=cache_dir, train_dataset=train_dataset, device=device,
        )
        self.explainer_cls = explainer_cls
        self.explain_kwargs = explain_kwargs
        self._init_explainer(explainer_cls, **explain_kwargs)

    def _init_explainer(self, cls: type, **explain_kwargs: Any):
        self.captum_explainer = cls(**explain_kwargs)
        if not isinstance(self.captum_explainer, self.explainer_cls):
            raise ValueError(f"Expected {self.explainer_cls}, but got {type(self.captum_explainer)}")

    def _process_targets(self, targets: Optional[Union[List[int], torch.Tensor]]):
        if targets is not None:
            # TODO: move validation logic outside at a later point
            validate_1d_tensor_or_int_list(targets)
            if isinstance(targets, list):
                targets = torch.tensor(targets)
            targets = targets.to(self.device)
        return targets

    def explain(
        self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None, **explain_fn_kwargs
    ) -> torch.Tensor:
        # Process inputs
        test = test.to(self.device)
        targets = self._process_targets(targets)

        if targets is not None:
            return self.captum_explainer.influence(inputs=(test, targets), **explain_fn_kwargs)
        else:
            return self.captum_explainer.influence(inputs=test, **explain_fn_kwargs)


class CaptumSimilarity(CaptumInfluence):
    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: str,
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        explainer_kwargs: Dict[str, Any],
    ):
        # extract and validate layer from kwargs
        self._layer: Union[List[str], str] = None
        self.layer = explainer_kwargs.get("layers", [])

        explainer_kwargs = {
            "module": model,
            "influence_src_dataset": train_dataset,
            "activation_dir": cache_dir,
            "model_id": model_id,
            "similarity_direction": "max",
            **explainer_kwargs,
        }

        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
            explainer_cls=SimilarityInfluence,
            explain_kwargs=explainer_kwargs,
        )

    @property
    def layer(self):
        return self._layer

    @layer.setter
    def layer(self, layers: Any):
        """
        Our wrapper only allows a single layer to be passed, while the Captum implementation allows multiple layers.
        Here, we validate if there is only a single layer passed.
        """
        if isinstance(layers, str):
            self._layer = layers
            return
        if len(layers) != 1:
            raise ValueError("A single layer shall be passed to the CaptumSimilarity explainer.")
        self._layer = layers[0]

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None, **kwargs: Any):
        # We might want to pass the top_k as an argument in some scenarios
        top_k = kwargs.get("top_k", len(self.train_dataset))

        topk_idx, topk_val = super().explain(test=test, top_k=top_k, **kwargs)[self.layer]
        inverted_idx = topk_idx.argsort()
        # Note to Galip: this is equivalent to your implementation (I checked the values)
        tda = torch.gather(topk_val, 1, inverted_idx)

        return tda
