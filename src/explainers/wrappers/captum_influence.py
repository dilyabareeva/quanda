from typing import Any, Callable, List, Optional, Union

import torch
from captum.influence import SimilarityInfluence  # type: ignore

from src.explainers.base import BaseExplainer
from src.explainers.utils import (
    explain_fn_from_explainer,
    self_influence_fn_from_explainer,
)
from src.utils.functions.similarities import cosine_similarity
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
        **explain_kwargs: Any,
    ):
        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
        )
        self.explainer_cls = explainer_cls
        self.explain_kwargs = explain_kwargs
        self._init_explainer(**explain_kwargs)

    def _init_explainer(self, **explain_kwargs: Any):
        self.captum_explainer = self.explainer_cls(**explain_kwargs)

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
    # TODO: incorporate SimilarityInfluence kwargs into init_kwargs
    """
    init_kwargs = signature(SimilarityInfluence.__init__).parameters.items()
    init_kwargs.append("replace_nan")
    explain_kwargs = signature(SimilarityInfluence.influence)
    si_kwargs = signature(SimilarityInfluence.selinfluence)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: str,
        train_dataset: torch.utils.data.Dataset,
        layers: Union[str, List[str]],
        similarity_metric: Callable = cosine_similarity,
        similarity_direction: str = "max",
        batch_size: int = 1,
        replace_nan: bool = False,
        device: Union[str, torch.device] = "cpu",
        **explainer_kwargs: Any,
    ):
        # extract and validate layer from kwargs
        self._layer: Optional[Union[List[str], str]] = None
        self.layer = layers

        # TODO: validate SimilarityInfluence kwargs
        explainer_kwargs.update(
            {
                "module": model,
                "influence_src_dataset": train_dataset,
                "activation_dir": cache_dir,
                "model_id": model_id,
                "layers": self.layer,
                "similarity_direction": similarity_direction,
                "similarity_metric": similarity_metric,
                "batch_size": batch_size,
                "replace_nan": replace_nan,
                **explainer_kwargs,
            }
        )

        super().__init__(
            model=model,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
            explainer_cls=SimilarityInfluence,
            **explainer_kwargs,
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
        top_k = kwargs.get("top_k", self.dataset_length)

        topk_idx, topk_val = super().explain(test=test, top_k=top_k, **kwargs)[self.layer]
        inverted_idx = topk_idx.argsort()
        # Note to Galip: this is equivalent to your implementation (I checked the values)
        tda = torch.gather(topk_val, 1, inverted_idx)

        return tda


def captum_similarity_explain(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    return explain_fn_from_explainer(
        explainer_cls=CaptumSimilarity,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        test_tensor=test_tensor,
        targets=explanation_targets,
        train_dataset=train_dataset,
        device=device,
        **kwargs,
    )


def captum_similarity_self_influence(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    **kwargs: Any,
) -> torch.Tensor:
    return self_influence_fn_from_explainer(
        explainer_cls=CaptumSimilarity,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        **kwargs,
    )
