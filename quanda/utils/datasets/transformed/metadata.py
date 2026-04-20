"""Dataset metadata classes."""

import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, TypeVar

import torch
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from quanda.utils.common import ds_len
from quanda.utils.datasets.dataset_handlers import (
    get_dataset_handler,
)

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
        for k, v in list(data_dict.items()):
            if isinstance(v, (DictConfig, ListConfig)):
                data_dict[k] = OmegaConf.to_container(v, resolve=True)

        with open(metadata_path, "w") as f:
            yaml.safe_dump(data_dict, f)

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

        handler = get_dataset_handler(dataset)
        self.mislabeling_labels = {
            i: self._poison(handler.get_label(dataset[i]))
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
    """Per-dataset metadata for LabelGroupingDataset.

    The class-to-group mapping itself lives in :class:`ClassMapping`, which
    is a shared artifact across train/val/eval datasets. This metadata only
    carries the dataset-specific transform indices.
    """

    def generate_indices(self, dataset: torch.utils.data.Dataset) -> List[int]:
        """Generate indices for transformation."""
        return super().generate_indices(dataset)

    def validate(self, dataset: torch.utils.data.Dataset):
        """Validate the metadata."""
        super().validate(dataset)


@dataclass
class ClassMapping:
    """Shared class-to-group mapping for :class:`LabelGroupingDataset`.

    Serialized as a standalone artifact so that multiple datasets (train,
    val, eval) can reference the same mapping by filename. Resolved from a
    config spec that is either a direct mapping (a ``Dict[int, int]``) or a
    file-backed reference specifying ``ctg_filename``, ``n_classes``,
    ``n_groups`` and an optional ``seed``.
    """

    class_to_group: Dict[int, int]
    n_classes: int
    n_groups: int
    seed: int = 42

    @classmethod
    def exists(cls, path: str, name: str) -> bool:
        """Check if a mapping file exists on disk."""
        return os.path.exists(os.path.join(path, name))

    def save(self, path: str, name: str) -> None:
        """Save the mapping to disk as YAML."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, name), "w") as f:
            yaml.safe_dump(
                {
                    "class_to_group": self.class_to_group,
                    "n_classes": self.n_classes,
                    "n_groups": self.n_groups,
                    "seed": self.seed,
                },
                f,
            )

    @classmethod
    def load(cls, path: str, name: str) -> "ClassMapping":
        """Load a mapping from disk."""
        with open(os.path.join(path, name), "r") as f:
            data = yaml.safe_load(f)
        return cls(
            class_to_group={
                int(k): int(v) for k, v in data["class_to_group"].items()
            },
            n_classes=int(data["n_classes"]),
            n_groups=int(data["n_groups"]),
            seed=int(data.get("seed", 42)),
        )

    @classmethod
    def _generate(
        cls, n_classes: int, n_groups: int, seed: int
    ) -> Dict[int, int]:
        rng = torch.Generator()
        rng.manual_seed(seed)
        return {
            i: int(torch.randint(n_groups, (1,), generator=rng).item())
            for i in range(n_classes)
        }

    @classmethod
    def resolve(
        cls,
        spec: dict,
        metadata_dir: str,
        load_meta_from_disk: bool,
    ) -> "ClassMapping":
        """Resolve a ``class_to_group`` config spec to a ``ClassMapping``.

        Spec forms:
          - ``{0: g0, 1: g1, ...}`` — direct mapping (integer keys).
          - ``{ctg_filename, n_classes, n_groups, seed?}`` — file-backed;
            load if exists, otherwise generate from ``seed`` and save.
        """
        if spec and all(isinstance(k, int) for k in spec.keys()):
            mapping = {int(k): int(v) for k, v in spec.items()}
            return cls(
                class_to_group=mapping,
                n_classes=len(mapping),
                n_groups=len(set(mapping.values())),
            )

        ctg_filename = spec["ctg_filename"]
        n_classes = int(spec["n_classes"])
        n_groups = int(spec["n_groups"])
        seed = int(spec.get("seed", 42))

        if load_meta_from_disk:
            if not cls.exists(metadata_dir, ctg_filename):
                raise FileNotFoundError(
                    f"Class mapping '{ctg_filename}' not found in "
                    f"{metadata_dir}. Re-run with "
                    f"load_meta_from_disk=False to regenerate it."
                )
            return cls.load(metadata_dir, ctg_filename)

        mapping = cls._generate(n_classes, n_groups, seed)
        instance = cls(
            class_to_group=mapping,
            n_classes=n_classes,
            n_groups=n_groups,
            seed=seed,
        )
        instance.save(metadata_dir, ctg_filename)
        return instance
