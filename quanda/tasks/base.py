from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import torch


class Task(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        explainer_cls: Optional[type] = None,
        expl_kwargs: Optional[dict] = None,
        model_id: Optional[str] = "0",
        cache_dir: str = "./cache",
        **kwargs,
    ):
        self.device: Union[str, torch.device]
        self.model = model

        # if model has device attribute, use it, otherwise use the default device
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

        self.model = model
        self.train_dataset = train_dataset
        self.expl_kwargs = expl_kwargs or {}
        if explainer_cls is not None:
            self.explainer = explainer_cls(
                model=self.model, train_dataset=train_dataset, model_id=model_id, cache_dir=cache_dir, **self.expl_kwargs
            )

    @abstractmethod
    def update(
        self,
        explanations: torch.Tensor,
        return_intermediate: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Used to perform the task with new data.

        Parameters
        ----------
        explanations: torch.Tensor
            The explanations.
        return_intermediate: bool
            Whether to return intermediate results.
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def explain_update(
        self,
        test_data: torch.Tensor,
        explanation_targets: torch.Tensor,
        explanations: torch.Tensor,
        return_intermediate: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Used to epxlain and perform the task with new data.

        Parameters
        ----------
        test_data: torch.Tensor
            The test data.
        explanation_targets: torch.Tensor
            The explanation targets.
        explanations: torch.Tensor
            The explanations.
        return_intermediate: bool
            Whether to return intermediate results.
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        None
        """
        if hasattr(self, "explainer"):
            raise NotImplementedError
        raise RuntimeError("No explainer is supplied to the task.")

    @abstractmethod
    def compute(self, *args: Any, **kwargs: Any) -> Any:
        """
        Used to compute the metric.

        Parameters
        ----------
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        Any
            The computed metric result dictionary.
        """

        raise NotImplementedError

    @abstractmethod
    def reset(self, *args: Any, **kwargs: Any):
        """
        Used to reset the metric.

        Parameters
        ----------
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        None
        """
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        """
        Used to load the metric state.

        Parameters
        ----------
        state_dict: dict
            The metric state dictionary.

        Returns
        -------
        None
        """

        raise NotImplementedError

    @abstractmethod
    def state_dict(self, *args: Any, **kwargs: Any) -> dict:
        """
        Used to get the metric state.

        Parameters
        ----------
        *args: Any
            Additional arguments.
        **kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        dict
            The metric state dictionary.
        """

        raise NotImplementedError
