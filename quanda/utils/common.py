import functools
from contextlib import contextmanager
from functools import reduce
from typing import Any, Callable, List, Mapping, Optional, Sized, Union

import torch.utils
import torch.utils.data


def _get_module_from_name(model: torch.nn.Module, layer_name: str) -> Any:
    """
    Simple helper function to get a module from a model by name.

    Parameters
    ----------
    model : torch.nn.Module
        The model to extract the module from.
    layer_name : str
        The name of the module to extract.

    Returns
    -------
    Any
        The module extracted from the model.
    """
    return reduce(getattr, layer_name.split("."), model)


def get_parent_module_from_name(model: torch.nn.Module, layer_name: str) -> Any:
    """
    Get the parent module of a module in a model by name.

    Parameters
    ----------
    model : torch.nn.Module
        The model to extract the module from.
    layer_name : str
        The name of the module to extract.

    Returns
    -------
    Any
        The module extracted from the model.
    """
    return reduce(getattr, layer_name.split(".")[:-1], model)


def make_func(func: Callable, func_kwargs: Optional[Mapping[str, Any]] = None, **kwargs) -> functools.partial:
    """
    A function for creating a partial function with the given arguments.

    Parameters
    ----------
    func : Callable
        The function to create a partial function from.
    func_kwargs : Optional[Mapping[str, Any]]
        Optional keyword arguments to fix for the function.

    Returns
    -------
    functools.partial
        The partial function with the given arguments.
    """
    if func_kwargs is not None:
        _func_kwargs = kwargs.copy()
        _func_kwargs.update(func_kwargs)
    else:
        _func_kwargs = kwargs

    return functools.partial(func, **_func_kwargs)


def cache_result(method):
    """Decorator to cache method results."""
    cache_attr = f"_{method.__name__}_cache"

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if cache_attr not in self.__dict__:
            self.__dict__[cache_attr] = method(self, *args, **kwargs)
        return self.__dict__[cache_attr]

    return wrapper


def class_accuracy(net: torch.nn.Module, loader: torch.utils.data.DataLoader, device: Union[str, torch.device] = "cpu"):
    """
    Return accuracy on a dataset given by the data loader.

    Parameters
    ----------
    net : torch.nn.Module
        The model to evaluate.
    loader : torch.utils.data.DataLoader
        The data loader to evaluate the model on.
    device : Union[str, torch.device], optional
        The device to evaluate the model on, by default "cpu".

    Returns
    -------
    float

    """
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
    """
    Helper to load pytorch models. This function attempts to find compatibility for
    loading models that were trained on different devices / with DataParallel but are
    being loaded in a different environment.

    Assumes that the model has been saved as a state_dict in some capacity. This can
    either be a single state dict, or a nesting dictionary which contains the model
    state_dict and other information.

    Parameters
    ----------
    model : torch.nn.Module
        The model for which to load a checkpoint
    path : str
        The filepath to the checkpoint
    device : Union[str, torch.device]
        The device to use.

    Returns
    -------
    float
        The learning rate.

    Notes
    -----
    The module state_dict is modified in-place.
    """
    if isinstance(device, str):
        device = torch.device(device)

    checkpoint = torch.load(path, map_location=device, weights_only=True)

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
    """
    Function to get a load_state_dict function that loads a model state dict

    Parameters
    ----------
    device : Union[str, torch.device]
        The device to load the model on.
    """

    def load_state_dict(model: torch.nn.Module, path: str) -> float:
        return _load_flexible_state_dict(model, path, device)

    return load_state_dict


@contextmanager
def default_tensor_type(device: Union[str, torch.device]):
    """
    Context manager to temporarily change the default tensor type.

    Parameters
    ----------
    device : Union[str, torch.device]
        The device to which the default tensor type should be set.

    Returns
    -------
    None

    """
    # Save the current default tensor type
    float_tensor = torch.FloatTensor([0.0])
    original_tensor_type = float_tensor.type()

    tensor = float_tensor.to(device)
    new_tensor_type = tensor.type()

    # Set the new tensor type
    torch.set_default_tensor_type(new_tensor_type)

    device_type = device.type if isinstance(device, torch.device) else device
    if "cuda" in device_type:
        torch.cuda.set_device(device)

    with torch.device(device):
        try:
            # Yield control back to the calling context
            yield
        finally:
            # Restore the original tensor type
            torch.set_default_tensor_type(original_tensor_type)


@contextmanager
def map_location_context(device: Union[str, torch.device]):
    """
    Context manager to temporarily change the map_location parameter of torch.load.

    Parameters
    ----------
    device: Union[str, torch.device]
        The device to which the tensors should be loaded.

    Returns
    -------
    None
    """
    original_load = torch.load

    # Custom function that wraps torch.load with a fixed map_location
    def load_with_map_location(f, *args, **kwargs):
        kwargs["map_location"] = device
        return original_load(f, *args, **kwargs)

    # Temporarily replace torch.load with our custom version
    torch.load = load_with_map_location
    try:
        yield  # Control returns to the code block within the `with` statement
    finally:
        # Restore the original torch.load function
        torch.load = original_load


def ds_len(dataset: torch.utils.data.Dataset) -> int:
    """
    Get the length of the dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to get the length of.

    Returns
    -------
    int
        The length of the dataset.
    """
    if isinstance(dataset, Sized):
        return len(dataset)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1)
    return len(dl)


def process_targets(targets: Union[List[int], torch.Tensor], device: Union[str, torch.device]) -> torch.Tensor:
    """
    Convert target labels to torch.Tensor and move them to the device.

    Parameters
    ----------
    targets : Optional[Union[List[int], torch.Tensor]], optional
        The target labels, either as a list or tensor.
    device: Union[str, torch.device]
        The device to use.

    Returns
    -------
    torch.Tensor or None
        The processed targets as a tensor, or None if no targets are provided.
    """
    if targets is not None:
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        targets = targets.to(device)
    return targets
