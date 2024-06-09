import functools
from functools import reduce
from typing import Any, Callable, Mapping, Optional

import torch
import torch.utils
import torch.utils.data

from utils.explain_wrapper import SelfInfluenceFunction


def _get_module_from_name(model: torch.nn.Module, layer_name: str) -> Any:
    return reduce(getattr, layer_name.split("."), model)


def _get_parent_module_from_name(model: torch.nn.Module, layer_name: str) -> Any:
    return reduce(getattr, layer_name.split(".")[:-1], model)


def make_func(func: Callable, func_kwargs: Mapping[str, ...] | None, **kwargs) -> functools.partial:
    """A function for creating a partial function with the given arguments."""
    if func_kwargs is not None:
        _func_kwargs = kwargs.copy()
        _func_kwargs.update(func_kwargs)
    else:
        func_kwargs = kwargs

    return functools.partial(func, **func_kwargs)


def get_self_influence_ranking(
    model: torch.nn.Module,
    model_id: str,
    cache_dir: Optional[str],
    training_data: torch.utils.data.Dataset,
    self_influence_fn: SelfInfluenceFunction,
    self_influence_fn_kwargs: Optional[dict] = None,
) -> torch.Tensor:
    size = len(training_data)
    self_inf = torch.zeros((size,))
    for i, (x, y) in enumerate(training_data):
        self_inf[i] = self_influence_fn(model, model_id, cache_dir, training_data, i)
    return self_inf.argsort()
