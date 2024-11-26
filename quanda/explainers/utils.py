"""Utility functions for explainer classes."""

from typing import Any, List, Optional, Union, Callable

import torch


def _init_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    checkpoints: Union[str, List[str]],
    train_dataset: torch.utils.data.Dataset,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    **kwargs,
):
    """Initialize an explainer.

    Parameters
    ----------
    explainer_cls : type
        The explainer class to initialize.
    model : torch.nn.Module
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        The path to the checkpoint(s) to load the model from.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints_load_func : Optional[Callable[..., Any]], optional
        The function to load the checkpoint(s), by default None.
    **kwargs : dict
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    explainer_cls
        Initialized explainer instance.

    """
    explainer = explainer_cls(
        model=model,
        checkpoints=checkpoints,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )
    return explainer


def explain_fn_from_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    test_tensor: torch.Tensor,
    checkpoints: Union[str, List[str]],
    train_dataset: torch.utils.data.Dataset,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    targets: Optional[Union[List[int], torch.Tensor]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Compute influence scores using the specified explainer class.

    Parameters
    ----------
    explainer_cls : type
        The explainer class to use for computing explanations.
    model : torch.nn.Module
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        The path to the checkpoint(s) to load the model from.
    test_tensor : torch.Tensor
        The test samples for which influence scores are computed.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints_load_func : Optional[Callable[..., Any]], optional
        The function to load the checkpoint(s), by default None.
    targets : Optional[Union[List[int], torch.Tensor]], optional
        Labels for the test samples. Defaults to None.
    **kwargs : dict
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the
        influence scores.

    """
    explainer = _init_explainer(
        explainer_cls=explainer_cls,
        model=model,
        checkpoints=checkpoints,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )

    return explainer.explain(test_tensor=test_tensor, targets=targets)


def self_influence_fn_from_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    checkpoints: Union[str, List[str]],
    train_dataset: torch.utils.data.Dataset,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    batch_size: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    """Compute self-influence scores using the specified explainer class.

    Parameters
    ----------
    explainer_cls : type
        The explainer class to use for computing explanations.
    model : torch.nn.Module
        The model to be used for the influence computation.
    checkpoints : Union[str, List[str]]
        The path to the checkpoint(s) to load the model from.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints_load_func : Optional[Callable[..., Any]], optional
        The function to load the checkpoint(s), by default None.
    batch_size : int, optional
        Batch size used for iterating over the dataset. Defaults to 1.
    **kwargs : dict
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        Self-influence scores for each datapoint in train_dataset.

    """
    explainer = _init_explainer(
        explainer_cls=explainer_cls,
        model=model,
        checkpoints=checkpoints,
        train_dataset=train_dataset,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )

    return explainer.self_influence(batch_size=batch_size)
