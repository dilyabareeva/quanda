import random

import torch
from torch.utils.data.dataset import Dataset

from utils.cache import IndicesCache as IC


class CorruptLabelDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        dataset_id: str,
        class_labels,
        classes,
        inverse_transform,
        cache_path="./datasets",
        p=0.3,
    ):
        super().__init__()
        self.dataset = dataset
        self.class_labels = class_labels
        self.classes = classes
        self.inverse_transform = inverse_transform
        self.p = p
        assert hasattr(dataset, "device")
        self.device = dataset.device

        if IC.exists(path=cache_path, file_id=f"{dataset_id}_corrupt_ids") and IC.exists(path=cache_path, file_id=f"{dataset_id}_corrupt_labels"):
            self.corrupt_indices = IC.load(path=cache_path, file_id=f"{dataset_id}_corrupt_ids")
            self.corrupt_labels = IC.load(path=cache_path, file_id=f"{dataset_id}_corrupt_labels")
        else:
            self.corrupt_indices = self.get_corrupt_sample_ids()
            IC.save(path=cache_path, file_id=f"{dataset_id}_corrupt_ids", indices=self.corrupt_indices, device=self.device)

            self.corrupt_labels = [self.corrupt_label(self.dataset[i][1]) for i in self.corrupt_indices]
            IC.save(path=cache_path, file_id=f"{dataset_id}_corrupt_labels", indices=self.corrupt_labels)

    def get_corrupt_sample_ids(self):
        torch.manual_seed(27)
        corrupt = torch.rand(len(self.dataset))
        return torch.where(corrupt < self.p)[0]

    def __getitem__(self, item):
        x, y_true = self.dataset[item]
        y = y_true
        if item in self.corrupt_indices:
            y = int(
                self.corrupt_labels[torch.squeeze((self.corrupt_indices == item).nonzero())]
            )  # TODO: not the most elegant solution
        return x, (y, y_true)

    def __len__(self):
        return len(self.dataset)

    def corrupt_label(self, y):
        classes = [cls for cls in self.classes if cls != y]
        random.seed(27)
        corrupted_class = random.choice(classes)
        return corrupted_class
