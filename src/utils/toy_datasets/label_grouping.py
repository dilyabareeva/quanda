from typing import Dict, Literal, Union

import torch

from src.utils.toy_datasets.base import ToyDataset

ClassToGroupLiterals = Literal["random"]


class GroupLabelDataset(ToyDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        # If isinstance(subset_idx,int): perturb this class with probability p,
        # if isinstance(subset_idx,List[int]): perturb datapoints with these indices with probability p
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
            subset_idx=None,  # apply with certainty, to all datapoints
        )
        self.n_classes = n_classes
        self.classes = list(range(n_classes))
        self.n_groups = n_groups
        self.groups = list(range(n_groups))
        if class_to_group == "random":
            # create a dictionary of class groups that assigns each class to a group
            self.class_to_group = {i: torch.randint(low=0, high=n_groups, generator=self.generator) for i in range(n_classes)}
        elif isinstance(class_to_group, dict):
            self._validate_class_to_group(class_to_group)
            self.class_to_group = class_to_group
        else:
            raise ValueError(f"Invalid class_to_group value: {class_to_group}")
        self.label_fn = lambda x: self.class_to_group[x]

    def _validate_class_to_group(self, class_to_group):
        assert len(class_to_group) == self.n_classes
        assert all([g in self.groups for g in self.class_to_group.values()])
