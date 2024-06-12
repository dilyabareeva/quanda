import random
from typing import Dict, Literal, Optional, Union

import torch
from torch.utils.data import Dataset

ClassToGroupLiterals = Literal["random"]


class GroupLabelDataset:
    def __init__(
        self,
        dataset: Dataset,
        n_classes: int = 10,
        n_groups: int = 2,
        class_to_group: Union[ClassToGroupLiterals, Dict[int, int]] = "random",
        seed: Optional[int] = 27,
        device: int = "cpu",
    ):
        self.dataset = dataset
        self.n_classes = n_classes
        self.classes = list(range(n_classes))
        self.n_groups = n_groups
        self.groups = list(range(n_groups))
        self.generator = torch.Generator(device=device)
        if class_to_group == "random":
            # create a dictionary of class groups that assigns each class to a group
            random.seed(seed)
            self.class_to_group = {i: random.choice(self.groups) for i in range(n_classes)}
        elif isinstance(class_to_group, dict):
            self.validate_class_to_group(class_to_group)
            self.class_to_group = class_to_group
        else:
            raise ValueError(f"Invalid class_to_group value: {class_to_group}")

    def validate_class_to_group(self, class_to_group):
        assert len(class_to_group) == self.n_classes
        assert all([g in self.groups for g in self.class_to_group.values()])

    def __getitem__(self, index):
        x, y = self.dataset[index]
        g = self.class_to_group[y]
        return x, g

    def get_subclass_label(self, index):
        _, y = self.dataset[index]
        return y

    def __len__(self):
        return len(self.dataset)
