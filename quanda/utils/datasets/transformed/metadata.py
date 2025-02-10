"""Dataset metadata classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Union, Literal, Type, TypeVar
import os
import torch
import yaml


from quanda.utils.common import ds_len

# Define a type variable bound to DatasetMetadata
T = TypeVar("T", bound="DatasetMetadata")


@dataclass
class DatasetMetadata(ABC):
    """Base class for dataset metadata."""

    p: float = 1.0
    seed: int = 42
    transform_indices: Optional[List[int]] = None
    cls_idx: Optional[int] = None
    rng: torch.Generator = field(init=False)

    def __post_init__(self):
        """Initialize the random generators."""
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.rang = random.Random(self.seed)

    def __getstate__(self):
        """Copy the object's state and remove the transient generators."""
        state = self.__dict__.copy()
        state.pop("rng", None)
        state.pop("rang", None)
        return state

    def __setstate__(self, state):
        """Restore the object's state."""
        # Restore the state and reinitialize the random generators
        self.__dict__.update(state)
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.rang = random.Random(self.seed)

    @classmethod
    def exists(cls, path: str, name: str) -> bool:
        """Check if metadata exists on disk."""
        metadata_path = os.path.join(path, name)
        return os.path.exists(metadata_path)

    def save(self, path: str, name: str) -> None:
        """Save metadata to disk."""
        os.makedirs(path, exist_ok=True)
        metadata_path = os.path.join(path, name)
        data_dict = self.__dict__.copy()
        data_dict.pop("rng")
        data_dict.pop("rang")

        with open(metadata_path, "w") as f:
            yaml.dump(data_dict, f)

    @classmethod
    def load(cls: Type[T], path: str, name: str) -> T:
        """Load metadata from disk.

        Parameters
        ----------
        path : str
            Directory path to load metadata from.
        name : str
            Name of the metadata file.

        Returns
        -------
        T
            Loaded metadata instance of the appropriate subclass.

        Raises
        ------
        FileNotFoundError
            If metadata file doesn't exist.

        """
        metadata_path = os.path.join(path, name)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata found at {metadata_path}")

        with open(metadata_path, "r") as f:
            data_dict = yaml.safe_load(f)
        return cls(**data_dict)

    @abstractmethod
    def validate(self, dataset: torch.utils.data.Dataset):
        """Validate the metadata."""
        if not 0 <= self.p <= 1:
            raise ValueError("Transformation probability must be in [0, 1]")
        if self.transform_indices is not None:
            if not all(
                0 <= int(idx) < ds_len(dataset)
                for idx in self.transform_indices
            ):
                raise ValueError("Invalid transform indices")

    @abstractmethod
    def generate_indices(self, dataset: torch.utils.data.Dataset) -> List[int]:
        """Generate indices for transformation."""
        if self.transform_indices is not None:
            return self.transform_indices

        dataset_length = ds_len(dataset)
        trans_idx = torch.rand(dataset_length, generator=self.rng) <= self.p

        if self.cls_idx is not None:
            trans_idx *= torch.tensor(
                [dataset[i][1] == self.cls_idx for i in range(dataset_length)],
                dtype=torch.bool,
            )

        self.transform_indices = torch.where(trans_idx)[0].tolist()

        return self.transform_indices


@dataclass
class SampleTransformationMetadata(DatasetMetadata):
    """Metadata for sample transformations."""

    n_classes: int = 10

    def generate_indices(self, dataset: torch.utils.data.Dataset) -> List[int]:
        """Generate indices for transformation."""
        return super().generate_indices(dataset)

    def validate(self, dataset: torch.utils.data.Dataset):
        """Validate the metadata."""
        super().validate(dataset)


@dataclass
class LabelFlippingMetadata(DatasetMetadata):
    """Metadata for flipping labels."""

    n_classes: int = 10
    mislabeling_labels: Optional[Dict[int, int]] = None

    def generate_indices(self, dataset: torch.utils.data.Dataset) -> List[int]:
        """Generate indices for transformation."""
        if self.mislabeling_labels is None:
            self.mislabeling_labels = self.generate_mislabeling_labels(dataset)
        self.transform_indices = list(self.mislabeling_labels.keys())
        return self.transform_indices

    def generate_mislabeling_labels(self, dataset) -> Dict[int, int]:
        """Generate mislabeling labels."""
        if self.mislabeling_labels is not None and not isinstance(
            self.mislabeling_labels, dict
        ):
            raise ValueError(
                f"mislabeling_labels should be a dictionary, received "
                f"{type(self.mislabeling_labels)}"
            )
        if self.mislabeling_labels is not None:
            self.transform_indices = list(self.mislabeling_labels.keys())
            return self.mislabeling_labels
        if self.transform_indices is None:
            self.transform_indices = super().generate_indices(dataset)
        self.mislabeling_labels = {
            i: self._poison(dataset[i][1])
            for i in range(len(dataset))
            if i in self.transform_indices
        }
        return self.mislabeling_labels

    def _poison(self, original_label):
        """Poisons labels."""
        label_arr = [i for i in range(self.n_classes) if original_label != i]
        label_idx = self.rang.randint(0, len(label_arr) - 1)
        return label_arr[label_idx]

    def validate(self, dataset: torch.utils.data.Dataset):
        """Validate the metadata."""
        super().validate(dataset)

        if self.mislabeling_labels is not None and not all(
            0 <= new_label < self.n_classes
            for new_label in self.mislabeling_labels.values()
        ):
            raise ValueError("Invalid mislabeling labels")


@dataclass
class LabelGroupingMetadata(DatasetMetadata):
    """Metadata for grouping classes."""

    n_classes: int = 10
    n_groups: int = 2
    class_to_group: Union[Literal["random"], Dict[int, int]] = "random"

    def generate_indices(self, dataset: torch.utils.data.Dataset) -> List[int]:
        """Generate indices for transformation."""
        return super().generate_indices(dataset)

    def generate_class_mapping(self) -> Dict[int, int]:
        """Generate a mapping from class to group."""
        self.class_to_group = {
            i: int(
                torch.randint(self.n_groups, (1,), generator=self.rng).item()
            )
            for i in range(self.n_classes)
        }
        return self.class_to_group

    def validate(self, dataset: torch.utils.data.Dataset):
        """Validate the metadata."""
        super().validate(dataset)
        if (
            isinstance(self.class_to_group, dict)
            and len(self.class_to_group) != self.n_classes
        ):
            raise ValueError(
                f"Length of class_to_group dictionary ("
                f"{len(self.class_to_group)}"
                f") does not match number of classes ({self.n_classes})"
            )
