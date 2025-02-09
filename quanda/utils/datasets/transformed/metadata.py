"""Dataset metadata classes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Union, Literal, Type, TypeVar
import os
import torch

from quanda.utils.common import ds_len

# Define a type variable bound to DatasetMetadata
T = TypeVar('T', bound='DatasetMetadata')


@dataclass
class DatasetMetadata(ABC):
    """Base class for dataset metadata."""
    p: float = 1.0
    seed: int = 42
    cls_idx: Optional[int] = None
    transform_indices: Optional[List[int]] = None
    rng: torch.Generator = field(init=False)

    def __post_init__(self):
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.rang = random.Random(self.seed)

    def __getstate__(self):
        # Copy the object's state and remove the transient generators
        state = self.__dict__.copy()
        state.pop("rng", None)
        state.pop("rang", None)
        return state

    def __setstate__(self, state):
        # Restore the state and reinitialize the random generators
        self.__dict__.update(state)
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)
        self.rang = random.Random(self.seed)

    @classmethod
    @abstractmethod
    def get_filename(cls) -> str:
        """Get the filename for saving/loading metadata."""
        pass

    @classmethod
    def exists(cls, path: str, name: str) -> bool:
        metadata_path = os.path.join(path, name)
        return os.path.exists(metadata_path)

    def save(self, path: str, name: str) -> None:
        os.makedirs(path, exist_ok=True)
        metadata_path = os.path.join(path, name)
        data_dict = self.__dict__.copy()
        data_dict.pop("rng")
        data_dict.pop("rang")
        torch.save(data_dict, metadata_path)

    @classmethod
    def load(cls: Type[T], path: str, name: str) -> T:
        """Load metadata from disk and ensure the return type is that of the subclass.

        Parameters
        ----------
        path : str
            Directory path to load metadata from.

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

        data_dict = torch.load(metadata_path)
        return cls(**data_dict)

    @abstractmethod
    def validate(self, dataset: torch.utils.data.Dataset):
        if not 0 <= self.p <= 1:
            raise ValueError("Transformation probability must be in [0, 1]")
        if self.cls_idx is not None and not 0 <= self.cls_idx < self.n_classes:
            raise ValueError("Invalid class index for transformation")
        if self.transform_indices is not None:
            if not all(
                    0 <= int(idx) < len(dataset) for idx in self.transform_indices):
                raise ValueError("Invalid transform indices")

    @abstractmethod
    def generate_indices(self, dataset: torch.utils.data.Dataset) -> List[int]:
        if self.transform_indices is not None:
            return self.transform_indices

        dataset_length = ds_len(dataset)
        trans_idx = torch.rand(dataset_length, generator=self.rng) <= self.p

        if self.cls_idx is not None:
            trans_idx *= torch.tensor(
                [dataset[i][1] == self.cls_idx for i in range(dataset_length)],
                dtype=torch.bool
            )

        return torch.where(trans_idx)[0].tolist()


@dataclass
class SampleTransformationMetadata(DatasetMetadata):
    n_classes: int = 10

    @classmethod
    def get_filename(cls) -> str:
        return "sample_transformation_metadata.pt"

    def generate_indices(self, dataset: torch.utils.data.Dataset) -> List[int]:
        return super().generate_indices(dataset)

    def validate(self, dataset: torch.utils.data.Dataset):
        super().validate(dataset)


@dataclass
class LabelFlippingMetadata(DatasetMetadata):
    n_classes: int = 10
    mislabeling_labels: Optional[Dict[str, int]] = None

    @classmethod
    def get_filename(cls) -> str:
        return "label_flipping_metadata.pt"

    def generate_indices(self, dataset: torch.utils.data.Dataset) -> List[int]:
        return super().generate_indices(dataset)

    def generate_mislabeling_labels(self, dataset) -> Dict[str, int]:
        if self.mislabeling_labels is not None:
            return self.mislabeling_labels

        return {
                str(i): self._poison(dataset[i][1])
                for i in range(len(dataset))
                if i in self.transform_indices
            }

    def _poison(self, original_label):
        """Poisons labels."""
        label_arr = [i for i in range(self.n_classes) if original_label != i]
        label_idx = self.rang.randint(0, len(label_arr) - 1)
        return label_arr[label_idx]

    def validate(self, dataset: torch.utils.data.Dataset):
        super().validate(dataset)
        if self.mislabeling_labels is not None and not isinstance(
                self.mislabeling_labels, dict):
            raise ValueError(
                f"mislabeling_labels should be a dictionary, received {type(self.mislabeling_labels)}"
            )
        if self.mislabeling_labels is not None and not all(
                0 <= new_label < self.n_classes
                for new_label in self.mislabeling_labels.values()):
            raise ValueError("Invalid mislabeling labels")


@dataclass
class LabelGroupingMetadata(DatasetMetadata):
    n_classes: int = 10
    n_groups: int = 2
    class_to_group: Union[Literal["random"], Dict[int, int]] = "random"

    @classmethod
    def get_filename(cls) -> str:
        return "label_grouping_metadata.pt"

    def generate_indices(self, dataset: torch.utils.data.Dataset) -> List[int]:
        return super().generate_indices(dataset)

    def generate_class_mapping(self) -> Dict[int, int]:
        return {
            i: torch.randint(self.n_groups, (1,), generator=self.rng).item()
            for i in range(self.n_classes)
        }

    def validate(self, dataset: torch.utils.data.Dataset):
        super().validate(dataset)
        if isinstance(self.class_to_group, dict) and len(self.class_to_group) != self.n_classes:
            raise ValueError(
                f"Length of class_to_group dictionary ({len(self.class_to_group)}) does not match number of classes ({self.n_classes})"
            )
