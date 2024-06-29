from typing import Dict, Literal, Union

import torch

from src.utils.datasets.transformed.base import TransformedDataset

ClassToGroupLiterals = Literal["random"]


class LabelGroupingDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        n_groups: int,
        seed: int = 27,
        device: str = "cpu",
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
            self.class_to_group = {i: self.rng.randint(0, self.n_groups - 1) for i in range(self.n_classes)}

        elif isinstance(class_to_group, dict):
            self._validate_class_to_group(class_to_group)
            self.class_to_group = class_to_group
            self.n_classes = len(self.class_to_group)
            self.n_groups = len(set(self.class_to_group.values()))
        else:
            raise ValueError(f"Invalid class_to_group value: {class_to_group}")
        self.label_fn = lambda x: self.class_to_group[x]

    def _validate_class_to_group(self, class_to_group):
        assert len(class_to_group) == self.n_classes
        assert all([g in self.groups for g in class_to_group.values()])
