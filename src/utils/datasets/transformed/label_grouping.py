import warnings
from typing import Callable, Dict, List, Literal, Optional, Union

import torch

from src.utils.datasets.transformed import TransformedDataset

ClassToGroupLiterals = Literal["random"]


class LabelGroupingDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        dataset_transform: Optional[Callable] = None,
        transform_indices: Optional[List] = None,
        seed: int = 42,
        device: str = "cpu",
        n_groups: Optional[int] = None,
        class_to_group: Union[ClassToGroupLiterals, Dict[int, int]] = "random",
    ):
        super().__init__(
            dataset=dataset,
            n_classes=n_classes,
            dataset_transform=dataset_transform,
            transform_indices=transform_indices,
            seed=seed,
            device=device,
            p=1.0,
            cls_idx=None,
        )

        if class_to_group == "random":
            if n_groups is None:
                raise ValueError("n_classes and n_groups must be specified when class_to_group is 'random'")

            self.n_classes = n_classes
            self.n_groups = n_groups

            self.class_to_group = {i: self.rng.randrange(self.n_groups) for i in range(self.n_classes)}

        elif isinstance(class_to_group, dict):
            if n_groups is not None:
                warnings.warn("Class-to-group assignment is used. n_groups parameter is ignored.")

            self.class_to_group = class_to_group
            self.n_classes = len(self.class_to_group)
            self.n_groups = len(set(self.class_to_group))

        else:
            raise ValueError(f"Invalid class_to_group value: {class_to_group}")

        self.classes = list(range(self.n_classes))
        self.groups = list(range(self.n_groups))
        self._validate_class_to_group()
        self.label_fn = lambda x: self.class_to_group[x]

    def _validate_class_to_group(self):
        if not len(self.class_to_group) == self.n_classes:
            raise ValueError(
                f"Length of class_to_group dictionary ({len(self.class_to_group)}) "
                f"does not match number of classes ({self.n_classes})"
            )
        if not all([g in self.groups for g in self.class_to_group.values()]):
            raise ValueError(f"Invalid group assignment in class_to_group: {self.class_to_group.values()}")
