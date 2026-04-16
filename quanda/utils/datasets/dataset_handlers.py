"""Dataset handler classes."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Union

import datasets  # type: ignore
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator  # type: ignore


class DatasetHandler(ABC):
    """Abstract base class for dataset handling."""

    @abstractmethod
    def process_batch(
        self, batch: Any, device: Union[str, torch.device]
    ) -> Tuple[Any, torch.Tensor]:
        """Process a batch of data and return model inputs and labels.

        Parameters
        ----------
        batch : Any
            A batch of data.
        device : Union[str, torch.device]
            Device to move the data to.

        Returns
        -------
        Tuple[Any, torch.Tensor]
            A tuple (inputs, labels) where inputs may be a tensor or dict,
            and labels is a tensor.

        """
        pass

    @abstractmethod
    def get_model_inputs(self, inputs: Any) -> Any:
        """Extract model inputs from the processed batch inputs.

        Parameters
        ----------
        inputs : Any
            Raw inputs from the dataset.

        Returns
        -------
        Any
            Model-ready inputs.

        """
        pass

    @abstractmethod
    def get_predictions(self, outputs: Any) -> torch.Tensor:
        """Extract predictions from model outputs.

        Parameters
        ----------
        outputs : Any
            Raw outputs from the model.

        Returns
        -------
        torch.Tensor
            The extracted predictions.

        """
        pass

    @abstractmethod
    def create_dataloader(
        self,
        dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a DataLoader for the dataset.

        Parameters
        ----------
        dataset : Union[torch.utils.data.Dataset, datasets.Dataset]
            The dataset to load.
        batch_size : int
            Batch size.
        shuffle : bool, optional
            Whether to shuffle the dataset (default is False).
        num_workers : int, optional
            Number of workers for data loading, by default 0.

        Returns
        -------
        DataLoader
            Configured DataLoader.

        """
        pass

    @abstractmethod
    def get_label(self, item: Any) -> Any:
        """Extract the label from a single dataset item.

        Parameters
        ----------
        item : Any
            A single item as returned by ``dataset[i]``.

        Returns
        -------
        Any
            The label associated with the item.

        """
        pass

    @abstractmethod
    def with_label(self, item: Any, label: Any) -> Any:
        """Return a copy of ``item`` with its label replaced.

        Parameters
        ----------
        item : Any
            A single item as returned by ``dataset[i]``.
        label : Any
            The replacement label.

        Returns
        -------
        Any
            Item with the label replaced, matching the input item's format.

        """
        pass


class TorchDatasetHandler(DatasetHandler):
    """Handler for PyTorch datasets."""

    def get_label(self, item: Tuple[Any, Any]) -> Any:
        """Extract the label from a ``(sample, label)`` tuple."""
        return item[1]

    def with_label(self, item: Tuple[Any, Any], label: Any) -> Tuple[Any, Any]:
        """Return a ``(sample, label)`` tuple with the label replaced."""
        return item[0], label

    def process_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        device: Union[str, torch.device],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of data.

        Parameters
        ----------
        batch : Tuple[torch.Tensor, torch.Tensor]
            A tuple of (inputs, labels).
        device : Union[str, torch.device]
            The device to move the tensors to.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The processed inputs and labels on the specified device.

        """
        inputs, labels = batch
        return inputs.to(device), labels.to(device)

    def get_model_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Extract model inputs from the processed batch inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            The processed batch inputs.

        Returns
        -------
        torch.Tensor
            The inputs to be passed to the model.

        """
        return inputs

    def get_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Extract predictions from model outputs.

        Parameters
        ----------
        outputs : torch.Tensor
            The model outputs.

        Returns
        -------
        torch.Tensor
            The extracted predictions.

        """
        return outputs.argmax(dim=-1)

    def create_dataloader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a DataLoader for the dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to load.
        batch_size : int
            The batch size to use.
        shuffle : bool, optional
            Whether to shuffle the data, by default False.
        num_workers : int, optional
            Number of workers for data loading, by default 0.

        Returns
        -------
        DataLoader
            Configured DataLoader.

        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context="fork" if num_workers > 0 else None,
        )


class HuggingFaceDatasetHandler(DatasetHandler):
    """Handler for HuggingFace datasets."""

    def get_label(self, item: Dict[str, Any]) -> Any:
        """Extract the label from a HuggingFace dict item."""
        return item["labels"]

    def with_label(self, item: Dict[str, Any], label: Any) -> Dict[str, Any]:
        """Return a HuggingFace dict item with ``labels`` replaced."""
        new_item = dict(item)
        existing = item.get("labels")
        if isinstance(existing, torch.Tensor):
            new_item["labels"] = torch.tensor(
                label, dtype=existing.dtype, device=existing.device
            )
        else:
            new_item["labels"] = label
        return new_item

    def process_batch(
        self, batch: Dict[str, torch.Tensor], device: Union[str, torch.device]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Process a batch of data from a HuggingFace dataset.

        Parameters
        ----------
        batch : Dict[str, torch.Tensor]
            The batch dictionary containing inputs and labels.
        device : Union[str, torch.device]
            The device to move the tensors to.

        Returns
        -------
        Tuple[Dict[str, torch.Tensor], torch.Tensor]
            The processed inputs dictionary and labels on the specified device.

        """
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)
        return inputs, labels

    def get_model_inputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Extract model inputs from the processed batch inputs.

        Parameters
        ----------
        inputs : Dict[str, torch.Tensor]
            The processed batch inputs dictionary.

        Returns
        -------
        Dict[str, torch.Tensor]
            The inputs to be passed to the model.

        """
        allowed_keys = {"input_ids", "attention_mask", "token_type_ids"}
        return {
            key: value for key, value in inputs.items() if key in allowed_keys
        }

    def get_predictions(self, outputs: Any) -> torch.Tensor:
        """Extract predictions from model outputs.

        Parameters
        ----------
        outputs : Any
            The model outputs.

        Returns
        -------
        torch.Tensor
            The extracted predictions.

        """
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return logits.argmax(dim=-1)

    def create_dataloader(
        self,
        dataset: datasets.Dataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
    ) -> DataLoader:
        """Create a DataLoader for the dataset.

        Parameters
        ----------
        dataset : datasets.Dataset
            The dataset to load.
        batch_size : int
            The batch size to use.
        shuffle : bool, optional
            Whether to shuffle the data, by default False.
        num_workers : int, optional
            Number of workers for data loading, by default 0.

        Returns
        -------
        DataLoader
            Configured DataLoader.

        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=default_data_collator,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context="fork" if num_workers > 0 else None,
        )


def get_dataset_handler(
    dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
) -> DatasetHandler:
    """Return the correct DatasetHandler for the given dataset.

    Parameters
    ----------
    dataset : Union[torch.utils.data.Dataset, datasets.Dataset]
        The dataset which is either a PyTorch Dataset or HuggingFace Dataset.

    Returns
    -------
    DatasetHandler
        A handler instance suited for the dataset.

    """
    inner = getattr(dataset, "dataset", None)
    if inner is not None and not isinstance(dataset, datasets.Dataset):
        if (
            isinstance(inner, datasets.Dataset)
            and "labels" not in inner.features
        ):
            return TorchDatasetHandler()
        return get_dataset_handler(inner)
    if isinstance(dataset, datasets.Dataset):
        if "labels" not in dataset.features:
            raise ValueError(
                "HuggingFace dataset must contain 'labels' key. "
                f"Available features: {list(dataset.features.keys())}"
            )
        return HuggingFaceDatasetHandler()
    elif isinstance(dataset, torch.utils.data.Dataset):
        return TorchDatasetHandler()

    supported = [torch.utils.data.Dataset, datasets.Dataset]
    raise ValueError(
        f"Unsupported dataset type: {type(dataset)}. "
        f"Expected one of: {supported}"
    )
