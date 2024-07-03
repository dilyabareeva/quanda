from inspect import signature
from typing import Any, List, Optional, Union

import torch


def _init_explainer(explainer_cls, model, model_id, cache_dir, train_dataset, device, **kwargs):
    # Python get explainer_cls expected init keyword arguments
    exp_init_kwargs = signature(explainer_cls.__init__)
    init_kwargs = {k: v for k, v in kwargs.items() if k in exp_init_kwargs.parameters}
    explainer = explainer_cls(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        device=device,
        **init_kwargs,
    )
    return explainer


def explain_fn_from_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    targets: Optional[Union[List[int], torch.Tensor]] = None,
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
    model_id: str,
    cache_dir: Optional[str],
    train_dataset: torch.utils.data.Dataset,
    device: Union[str, torch.device],
    batch_size: Optional[int] = 32,
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

    return explainer.self_influence(batch_size=batch_size)
