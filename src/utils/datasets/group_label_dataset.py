import random
import warnings
from typing import Dict, Literal, Optional, Sized, Union

import torch
from torch.utils.data import Dataset

ClassToGroupLiterals = Literal["random"]


class GroupLabelDataset(Dataset):

    def __init__(
        self,
        dataset: Dataset,
        n_classes: Optional[int] = None,
        n_groups: Optional[int] = None,
        class_to_group: Union[ClassToGroupLiterals, Dict[int, int]] = "random",
        seed: Optional[int] = 27,
        device: str = "cpu",
    ):
        self.dataset = dataset
        self.generator = torch.Generator(device=device)

        if class_to_group == "random":

            if (n_classes is None) or (n_groups is None):
                raise ValueError("n_classes and n_groups must be specified when class_to_group is 'random'")

            self.n_classes = n_classes
            self.n_groups = n_groups

            # create a dictionary of class groups that assigns each class to a group
            random.seed(seed)

            self.class_to_group = {i: random.randrange(self.n_groups) for i in range(self.n_classes)}

        elif isinstance(class_to_group, dict):

            if (n_classes is not None) or (n_groups is not None):
                warnings.warn("Class-to-group assignment is used. n_classes or n_groups parameters are ignored.")

            self.class_to_group = class_to_group
            self.n_classes = len(self.class_to_group)
            self.n_groups = len(set(self.class_to_group.values()))

        else:

            raise ValueError(f"Invalid class_to_group value: {class_to_group}")

        self.classes = list(range(self.n_classes))
        self.groups = list(range(self.n_groups))
        self.validate_class_to_group()

    def validate_class_to_group(self):
        if not len(self.class_to_group) == self.n_classes:
            raise ValueError(
                f"Length of class_to_group dictionary ({len(self.class_to_group)}) "
                f"does not match number of classes ({self.n_classes})"
            )
        if not all([g in self.groups for g in self.class_to_group.values()]):
            raise ValueError(f"Invalid group assignment in class_to_group: {self.class_to_group.values()}")

    def __getitem__(self, index):
        x, y = self.dataset[index]
        g = self.class_to_group[y.item()]
        return x, g

    def get_subclass_label(self, index):
        _, y = self.dataset[index]
        return y

    def __len__(self):
        if isinstance(self.dataset, Sized):
            return len(self.dataset)
        dl = torch.utils.data.DataLoader(self.dataset, batch_size=1)
        return len(dl)
