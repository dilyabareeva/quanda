from abc import ABC, abstractmethod
from typing import List, Union, Optional, Callable, Any

import lightning as L
import torch

from quanda.utils.common import cache_result, ds_len, get_load_state_dict_func
from quanda.utils.datasets import OnDeviceDataset


class Explainer(ABC):
    """
    Base class for explainer wrappers. Defines the interface that all
    explainer classes must implement.

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        checkpoints: Union[str, List[str]],
        train_dataset: torch.utils.data.Dataset,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        **kwargs,
    ):
        """Initializer for the `Explainer` class.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        checkpoints : Union[str, List[str]]
            Path to the checkpoint file(s) to be used for the attribution computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, by default None.
        **kwargs : dict
            Additional keyword arguments passed to the explainer.
        """
        self.device: Union[str, torch.device]
        self.model = model
        self.checkpoints = checkpoints if isinstance(checkpoints, List) else [checkpoints]

        # if model has device attribute, use it, otherwise use the default device
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

        if checkpoints_load_func is None:
            self.checkpoints_load_func = get_load_state_dict_func(self.device)
        else:
            self.checkpoints_load_func = checkpoints_load_func

        # if dataset return samples not on device, move them to device
        if train_dataset[0][0].device != self.device:
            train_dataset = OnDeviceDataset(train_dataset, self.device)

        self.train_dataset = train_dataset

    @abstractmethod
    def explain(
        self,
        test_tensor: torch.Tensor,
        targets: Union[List[int], torch.Tensor],
    ) -> torch.Tensor:
        """
        Abstract method for computing influence scores for the test samples.

        Parameters
        ----------
        test_tensor : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Union[List[int], torch.Tensor]
            Labels for the test samples.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
        """
        raise NotImplementedError

    @cache_result
    def self_influence(self, batch_size: int = 32) -> torch.Tensor:
        """
        Compute self-influence scores for the training dataset by explaining the dataset one by one.

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
                test_tensor=x.to(self.device), targets=y.to(self.device)
            )
            influences[i : i + batch_size] = explanations.diag(diagonal=i)

        return influences

    def load_last_checkpoint(self):
        """
        Load the model from the checkpoint file.

        Parameters
        ----------
        checkpoint : str
            Path to the checkpoint file.
        """
        self.checkpoints_load_func(self.model, self.checkpoints[-1])
