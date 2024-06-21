from typing import Dict, List, Optional, Union

import torch


def explain_fn_from_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    targets: Optional[Union[List[int], torch.Tensor]] = None,
    init_kwargs: Optional[Dict] = None,
    explain_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    init_kwargs = init_kwargs or {}
    explain_kwargs = explain_kwargs or {}

    explainer = explainer_cls(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        **init_kwargs,
    )
    return explainer.explain(test=test_tensor, targets=targets, **explain_kwargs)


def self_influence_fn_from_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    batch_size: Optional[int] = 32,
    init_kwargs: Optional[Dict] = None,
    explain_kwargs: Optional[Dict] = None,
) -> torch.Tensor:
    init_kwargs = init_kwargs or {}
    explain_kwargs = explain_kwargs or {}

    explainer = explainer_cls(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        **init_kwargs,
    )
    return explainer.self_influence(batch_size=batch_size, **explain_kwargs)
