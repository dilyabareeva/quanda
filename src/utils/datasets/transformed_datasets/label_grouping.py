from typing import Dict, Literal, Union

import torch

from src.utils.datasets.transformed_datasets.base import TransformedDataset

ClassToGroupLiterals = Literal["random"]


class GroupLabelDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        seed: int = 42,
        device: str = "cpu",
        n_groups: int = 2,
        class_to_group: Union[ClassToGroupLiterals, Dict[int, int]] = "random",
    ):

        super().__init__(
            dataset=dataset,
            n_classes=n_classes,
            seed=seed,
            device=device,
            p=1.0,
            cls_idx=None,  # apply to all datapoints with certainty
        )
        self.n_classes = n_classes
        self.classes = list(range(n_classes))
        self.n_groups = n_groups
        self.groups = list(range(n_groups))
        if class_to_group == "random":
            # create a dictionary of class groups that assigns each class to a group
            group_assignments = [self.rng.randint(0, n_groups - 1) for _ in range(n_classes)]
            self.class_to_group = {}
            for i in range(n_classes):
                self.class_to_group[i] = group_assignments[i]

        elif isinstance(class_to_group, dict):
            self._validate_class_to_group(class_to_group)
            self.class_to_group = class_to_group
        else:
            raise ValueError(f"Invalid class_to_group value: {class_to_group}")
        self.label_fn = lambda x: self.class_to_group[x]

    def _validate_class_to_group(self, class_to_group):
        assert len(class_to_group) == self.n_classes
        assert all([g in self.groups for g in self.class_to_group.values()])
