import random
from typing import Callable, List

import torch
from torch.utils.data.dataset import Dataset

from utils.cache import IndicesCache as IC


class MarkDataset(Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dataset_id: str,
        class_labels: List[int],
        inverse_transform: Callable = lambda x: x,
        cache_path: str = "./datasets",
        p: float = 0.3,
        cls_to_mark: int = 2,
        only_train: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.dataset_id = dataset_id
        self.inverse_transform = inverse_transform
        self.class_labels = class_labels
        self.only_train = only_train
        self.cls_to_mark = cls_to_mark
        self.mark_prob = p

        if IC.exists(path=cache_path, file_id=f"{dataset_id}_mark_ids"):
            self.mark_indices = IC.load(path="./datasets", file_id=f"{dataset_id}_mark_ids")
        else:
            self.mark_indices = self.get_mark_sample_ids()
            IC.save(path=cache_path, file_id=f"{dataset_id}_mark_ids")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if index in self.mark_indices:
            return self.mark_image(x), y
        else:
            return x, y

    def get_mark_sample_ids(self):
        cls_labels = [data[1] for data in self.dataset if data[1] in self.cls_to_mark]
        n_marked = int(len(cls_labels) * self.mark_prob)
        random.seed(27)  # TODO: check best practices for setting seed
        indices = random.sample(cls_labels, n_marked)
        return torch.tensor(indices, dtype=torch.int)

    def mark_image_contour(self, x):
        x = self.inverse_transform(x)
        # TODO: make controur, middle square and combined masks a constant somewhere else
        mask = torch.zeros_like(x[0])
        mask[:2, :] = 1.0
        mask[-2:, :] = 1.0
        mask[:, -2:] = 1.0
        mask[:, :2] = 1.0
        x[0] = torch.ones_like(x[0]) * mask + x[0] * (1 - mask)
        if x.shape[0] > 1:
            x[1:] = torch.zeros_like(x[1:]) * mask + x[1:] * (1 - mask)

        return self.dataset.transform(x.numpy().transpose(1, 2, 0))

    def mark_image_middle_square(self, x):
        x = self.inverse_transform(x)
        mask = torch.zeros_like(x[0])
        mid = int(x.shape[-1] / 2)
        mask[(mid - 4) : (mid + 4), (mid - 4) : (mid + 4)] = 1.0
        x[0] = torch.ones_like(x[0]) * mask + x[0] * (1 - mask)
        if x.shape[0] > 1:
            x[1:] = torch.zeros_like(x[1:]) * mask + x[1:] * (1 - mask)
        return self.dataset.transform(x.numpy().transpose(1, 2, 0))

    def mark_image(self, x):
        x = self.inverse_transform(x)
        mask = torch.zeros_like(x[0])
        mid = int(x.shape[-1] / 2)
        mask[mid - 3 : mid + 3, mid - 3 : mid + 3] = 1.0
        mask[:2, :] = 1.0
        mask[-2:, :] = 1.0
        mask[:, -2:] = 1.0
        mask[:, :2] = 1.0
        x[0] = torch.ones_like(x[0]) * mask + x[0] * (1 - mask)
        if x.shape[0] > 1:
            x[1:] = torch.zeros_like(x[1:]) * mask + x[1:] * (1 - mask)
        return self.dataset.transform(x.numpy().transpose(1, 2, 0))
