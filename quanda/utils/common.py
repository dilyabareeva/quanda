import functools
from functools import reduce
from typing import Any, Callable, Mapping, Optional, Union

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


# Taken directly from Captum with minor changes
# (required because Captum's Arnoldi Influence Function does not allow to specify device)
def _load_flexible_state_dict(model: torch.nn.Module, path: str, device: Union[str, torch.device]) -> float:
    r"""
    Helper to load pytorch models. This function attempts to find compatibility for
    loading models that were trained on different devices / with DataParallel but are
    being loaded in a different environment.

    Assumes that the model has been saved as a state_dict in some capacity. This can
    either be a single state dict, or a nesting dictionary which contains the model
    state_dict and other information.

    Args:

        model (torch.nn.Module): The model for which to load a checkpoint
        path (str): The filepath to the checkpoint

    The module state_dict is modified in-place, and the learning rate is returned.
    """
    if isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(path, map_location=device)

    learning_rate = checkpoint.get("learning_rate", 1.0)

    if "module." in next(iter(checkpoint)):
        if isinstance(model, torch.nn.DataParallel):
            model.load_state_dict(checkpoint)
        else:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(checkpoint)
            model = model.module
    else:
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
            model.load_state_dict(checkpoint)
            model = torch.nn.DataParallel(model)
        else:
            model.load_state_dict(checkpoint)

    return learning_rate


def get_load_state_dict_func(device: Union[str, torch.device]):
    def load_state_dict(model: torch.nn.Module, path: str) -> float:
        return _load_flexible_state_dict(model, path, device)

    return load_state_dict