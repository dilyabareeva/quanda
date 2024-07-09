import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

import torch
from captum.influence import SimilarityInfluence  # type: ignore

from src.explainers.base import BaseExplainer
from src.explainers.utils import (
    explain_fn_from_explainer,
    self_influence_fn_from_explainer,
)
from src.utils.functions.similarities import cosine_similarity


class CaptumInfluence(BaseExplainer, ABC):

    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        explainer_cls: type,
        explain_kwargs: Any,
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

    @abstractmethod
    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor:
        """Comment for Galip and Niklas: We are now expecting explicit declaration of
        explain method keyword arguments in specific explainer class __init__ methods.
        Right now the only such keyword argument is `top_k` for CaptumSimilarity, which we
        anyway overwrite with the dataset length."""
        raise NotImplementedError


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

        if device != "cpu":
            warnings.warn("CaptumSimilarity explainer only supports CPU devices. Setting device to 'cpu'.")
            device = "cpu"

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
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
            explainer_cls=SimilarityInfluence,
            explain_kwargs=explainer_kwargs,
        )

        # explicitly specifying explain method kwargs as instance attributes
        self.top_k = self.dataset_length

        if "top_k" in explainer_kwargs:
            warnings.warn("top_k is not supported by CaptumSimilarity explainer. Ignoring the argument.")

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

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):

        test = test.to(self.device)

        if targets is not None:
            self._process_targets(targets=targets)
            warnings.warn("CaptumSimilarity explainer does not support target indices. Ignoring the argument.")

        topk_idx, topk_val = self.captum_explainer.influence(inputs=test, top_k=self.top_k)[self.layer]
        _, inverted_idx = topk_idx.sort()
        return torch.gather(topk_val, 1, inverted_idx)


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
