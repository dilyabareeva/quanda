import functools
from functools import reduce
from typing import Any, Callable, Mapping, Optional, Union

import torch
import torch.utils
import torch.utils.data


def _get_module_from_name(model: torch.nn.Module, layer_name: str) -> Any:
    return reduce(getattr, layer_name.split("."), model)


def get_parent_module_from_name(model: torch.nn.Module, layer_name: str) -> Any:
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


def class_accuracy(net: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str = "cpu"):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


def auc(x: torch.Tensor, max: Optional[Union[int, torch.Tensor]] = None):
    if max is None:
        max = x.max()
    return x.mean() / max


def cumsum(x: torch.Tensor):
    return [x[:i].sum() for i in range(x.shape[0])]
