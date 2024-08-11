from typing import Any, List, Optional, Union

import torch


def _init_explainer(explainer_cls, model, model_id, cache_dir, train_dataset, device, **kwargs):
    explainer = explainer_cls(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        **kwargs,
    )
    return explainer


def explain_fn_from_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    targets: Optional[Union[List[int], torch.Tensor]] = None,
    cache_dir: Optional[str] = None,
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> torch.Tensor:
    explainer = _init_explainer(
        explainer_cls=explainer_cls,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        **kwargs,
    )

    return explainer.explain(test=test_tensor, targets=targets)


def self_influence_fn_from_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    self_influence_kwargs: dict,
    cache_dir: Optional[str] = None,
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> torch.Tensor:
    explainer = _init_explainer(
        explainer_cls=explainer_cls,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        **kwargs,
    )

    return explainer.self_influence(**self_influence_kwargs)
