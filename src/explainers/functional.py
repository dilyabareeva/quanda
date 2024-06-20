from typing import Dict, List, Optional, Protocol, Union

import torch

from src.explainers.base_explainer import BaseExplainer
from src.explainers.wrappers.captum_influence import CaptumSimilarity


class ExplainFunc(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        test_tensor: torch.Tensor,
        explanation_targets: Optional[Union[List[int], torch.Tensor]],
        train_dataset: torch.utils.data.Dataset,
        explain_kwargs: Dict,
        init_kwargs: Dict,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        pass


def explain_fn_from_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    test_tensor: torch.Tensor,
    explanation_targets: Optional[Union[List[int], torch.Tensor]],
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    init_kwargs: Optional[Dict] = {},
    explain_kwargs: Optional[Dict] = {},
) -> torch.Tensor:
        explainer = explainer_cls(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
            explainer_kwargs=init_kwargs,
        )
        return explainer.explain(test=test_tensor, **explain_kwargs)


def explainer_self_influence_interface(
    explainer_cls: type,
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    train_dataset: torch.utils.data.Dataset,
    init_kwargs: Dict,
    device: Union[str, torch.device],
) -> torch.Tensor:
    explainer = explainer_cls(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        explainer_kwargs=init_kwargs,
    )
    return explainer.self_influence()


def captum_similarity_explain(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    test_tensor: torch.Tensor,
    explanation_targets: Optional[Union[List[int], torch.Tensor]],
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    init_kwargs: Optional[Dict] = {},
    explain_kwargs: Optional[Dict] = {},
) -> torch.Tensor:
    return explain_fn_from_explainer(
        explainer_cls=CaptumSimilarity,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        test_tensor=test_tensor,
        explanation_targets=explanation_targets,
        train_dataset=train_dataset,
        device=device,
        init_kwargs=init_kwargs,
        explain_kwargs=explain_kwargs,
    )


def captum_similarity_self_influence_ranking(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    train_dataset: torch.utils.data.Dataset,
    init_kwargs: Dict,
    device: Union[str, torch.device],
) -> torch.Tensor:
    return explainer_self_influence_interface(
        explainer_cls=CaptumSimilarity,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        init_kwargs=init_kwargs,
    )
