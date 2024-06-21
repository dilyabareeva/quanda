import functools
from functools import reduce
from typing import Any, Callable, Mapping, Optional

import torch
import torch.utils
import torch.utils.data


def _get_module_from_name(model: torch.nn.Module, layer_name: str) -> Any:
    return reduce(getattr, layer_name.split("."), model)


def _get_parent_module_from_name(model: torch.nn.Module, layer_name: str) -> Any:
    return reduce(getattr, layer_name.split(".")[:-1], model)


def make_func(func: Callable, func_kwargs: Optional[Mapping[str, Any]] = None, **kwargs) -> functools.partial:
    """A function for creating a partial function with the given arguments."""
    if func_kwargs is not None:
        _func_kwargs = kwargs.copy()
        _func_kwargs.update(func_kwargs)
    else:
        _func_kwargs = kwargs

    return functools.partial(func, **_func_kwargs)


def cache_result(method):
    cache_attr = f"_{method.__name__}_cache"

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if cache_attr not in self.__dict__:
            self.__dict__[cache_attr] = method(self, *args, **kwargs)
        return self.__dict__[cache_attr]

    return wrapper
