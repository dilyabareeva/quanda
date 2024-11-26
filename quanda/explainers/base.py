from abc import ABC, abstractmethod
from typing import List, Union

import lightning as L
import torch
from datasets import Dataset, DatasetDict  # type: ignore

from quanda.utils.common import cache_result, ds_len
from quanda.utils.datasets import OnDeviceDataset


class Explainer(ABC):
    """
    Base class for explainer wrappers. Defines the interface that all
    explainer classes must implement.

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        **kwargs,
    ):
        """Initializer for the `Explainer` class.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        **kwargs : dict
            Additional keyword arguments passed to the explainer.
        """
        self.device: Union[str, torch.device]
        self.model = model

        # if model has device attribute, use it, otherwise use the default device
        if next(model.parameters(), None) is not None:
            self.device = next(model.parameters()).device
        else:
            self.device = torch.device("cpu")

        # if dataset return samples not on device, move them to device
        # TODO: fix this
        if isinstance(train_dataset, (Dataset, DatasetDict)):
            pass
        elif train_dataset[0][0].device != self.device:
            train_dataset = OnDeviceDataset(train_dataset, self.device)

        self.train_dataset = train_dataset

    @abstractmethod
    def explain(self, test_tensor: Union[torch.Tensor, DatasetDict], targets: Union[List[int], torch.Tensor]) -> torch.Tensor:
        """
        Abstract method for computing influence scores for the test samples.

        Parameters
        ----------
        test_tensor : Union[torch.Tensor, DatasetDict]
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
        # Pre-allocate memory for influences, because torch.cat is slow
        influences = torch.empty((ds_len(self.train_dataset),), device=self.device)
        ldr = torch.utils.data.DataLoader(self.train_dataset, shuffle=False, batch_size=batch_size)
        batch_size = min(batch_size, ds_len(self.train_dataset))

        for i, batch in zip(range(0, ds_len(self.train_dataset), batch_size), ldr):
            inputs, targets = self.extract_batch(batch)
            inputs = self.move_to_device(inputs, self.device)
            targets = targets.to(self.device)
            explanations = self.explain(test_tensor=inputs, targets=targets)
            influences[i : i + len(targets)] = explanations.diag()

        return influences

    def move_to_device(self, data, device):
        if isinstance(data, DatasetDict):
            return {k: v.to(device) if torch.is_tensor(v) else v for k, v in data.items()}  # TODO: Validate this
        elif torch.is_tensor(data):
            return data.to(device)
        else:
            return data

    def extract_batch(self, batch):
        """
        Extract inputs and targets from a batch.
        """
        if isinstance(batch, DatasetDict):
            if "labels" in batch:
                targets = batch["labels"]
                inputs = {k: v for k, v in batch.items() if k != "labels"}
            elif "label" in batch:
                targets = batch["label"]
                inputs = {k: v for k, v in batch.items() if k != "label"}
            else:
                raise ValueError("Batch dict does not contain 'labels' or 'label'")
        else:
            *inputs, targets = batch
            if len(inputs) == 1:
                inputs = inputs[0]
        return inputs, targets
