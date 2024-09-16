from typing import Any, List, Optional, Union

import torch


def _init_explainer(explainer_cls, model, model_id, cache_dir, train_dataset, **kwargs):
    """
    Helper function to initialize an explainer.

    Parameters
    ----------
    explainer_cls : type
        The explainer class to initialize.
    model : torch.nn.Module
        The model to be used for the influence computation.
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    cache_dir : str
        Directory for caching results. Defaults to "./cache".
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    **kwargs : dict
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    explainer_cls
        Initialized explainer instance.
    """
    explainer = explainer_cls(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        **kwargs,
    )
    return explainer


def explain_fn_from_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    test_tensor: torch.Tensor,
    train_dataset: torch.utils.data.Dataset,
    targets: Optional[Union[List[int], torch.Tensor]] = None,
    cache_dir: str = "./cache",
    model_id: Optional[str] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Compute influence scores for the test samples using the specified explainer class.

    Parameters
    ----------
    explainer_cls : type
        The explainer class to use for computing explanations.
    model : torch.nn.Module
        The model to be used for the influence computation.
    test_tensor : torch.Tensor
        The test samples for which influence scores are computed.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    targets : Optional[Union[List[int], torch.Tensor]], optional
        Labels for the test samples. Defaults to None.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
    **kwargs : dict
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
    """
    explainer = _init_explainer(
        explainer_cls=explainer_cls,
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        **kwargs,
    )

    return explainer.explain(test=test_tensor, targets=targets)


def self_influence_fn_from_explainer(
    explainer_cls: type,
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    cache_dir: str = "./cache",
    model_id: Optional[str] = None,
    batch_size: int = 1,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Compute self-influence scores using the specified explainer class.

    Parameters
    ----------
    explainer_cls : type
        The explainer class to use for computing explanations.
    model : torch.nn.Module
        The model to be used for the influence computation.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    cache_dir : str, optional
        Directory for caching results. Defaults to "./cache".
    model_id : Optional[str], optional
        Identifier for the model. Defaults to None.
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
        model_id=model_id,
        cache_dir=cache_dir,
        train_dataset=train_dataset,
        **kwargs,
    )

    return explainer.self_influence(batch_size=batch_size)
