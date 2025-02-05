"""Base class for explainers."""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Callable, Any

import lightning as L
import torch
from datasets import Dataset, DatasetDict  # type: ignore

from quanda.utils.common import (
    cache_result,
    ds_len,
    get_load_state_dict_func,
    load_last_checkpoint,
)
from quanda.utils.datasets import OnDeviceDataset
from quanda.utils.tasks import TaskLiterals


class Explainer(ABC):
    """Base class for explainer wrappers.

    Defines the interface that all explainer classes must implement.

    """

    accepted_tasks: List[TaskLiterals] = []

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        **kwargs,
    ):
        """Initialize the `Explainer` class.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
            Possible options: "image_classification", "text_classification",
            "causal_lm".
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        **kwargs : dict
            Additional keyword arguments passed to the explainer.

        """
        if task not in self.accepted_tasks:
            raise ValueError(
                f"Task {task} not supported by this explainer. "
                f"Supported tasks: {self.accepted_tasks}"
            )

        self.device: Union[str, torch.device]
        self.model = model

        # if model has device attribute, use it, otherwise use the default
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

        if checkpoints_load_func is None:
            self.checkpoints_load_func = get_load_state_dict_func(self.device)
        else:
            self.checkpoints_load_func = checkpoints_load_func

        if checkpoints is None:
            self.checkpoints = []
        else:
            self.checkpoints = (
                checkpoints if isinstance(checkpoints, List) else [checkpoints]
            )

        # if dataset return samples not on device, move them to device
        # TODO: fix this
        if isinstance(train_dataset, (Dataset, DatasetDict)):
            pass
        elif train_dataset[0][0].device != self.device:
            train_dataset = OnDeviceDataset(train_dataset, self.device)

        self.train_dataset = train_dataset

    @abstractmethod
    def explain(
        self,
        test_data: Any,
        targets: Any,
    ) -> torch.Tensor:
        """Abstract method for computing influence scores for the test samples.

        Parameters
        ----------
        test_data : Any
            Test samples for which influence scores are computed.
        targets : Any
            Labels for the test samples.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing
            the influence scores.

        """
        raise NotImplementedError

    @cache_result
    def self_influence(self, batch_size: int = 32) -> torch.Tensor:
        """Compute self-influence scores by explaining one by one.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for iterating over the dataset. Defaults to 32.

        Returns
        -------
        torch.Tensor
            Self-influence score for each datapoint in the training dataset.

        """
        # Pre-allcate memory for influences, because torch.cat is slow
        influences = torch.empty(
            (ds_len(self.train_dataset),), device=self.device
        )
        ldr = torch.utils.data.DataLoader(
            self.train_dataset, shuffle=False, batch_size=batch_size
        )
        batch_size = min(batch_size, ds_len(self.train_dataset))

        for i, (x, y) in zip(
            range(0, ds_len(self.train_dataset), batch_size), ldr
        ):
            explanations = self.explain(
                test_data=x.to(self.device), targets=y.to(self.device)
            )
            influences[i : i + batch_size] = explanations.diag(diagonal=i)

        return influences

    def load_last_checkpoint(self):
        """Load the model from the checkpoint file.

        Parameters
        ----------
        checkpoint : str
            Path to the checkpoint file.

        """
        load_last_checkpoint(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
        )
