from typing import Callable, List, Optional, Union

import torch
from torch.utils.data.dataset import Dataset

from utils.cache import TensorCache as IC
from utils.transforms import mark_image_contour_and_square


class SampleTransformDataset(Dataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        dataset_id: str,
        class_labels: List[int],
        inverse_transform: Callable = lambda x: x,
        cache_path: str = "./datasets",
        p: float = 0.3,
        cls_to_mark: int = 2,
        mark_fn: Optional[Union[Callable, str]] = None,
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
        if mark_fn is not None:
            self.mark_image = mark_fn
        else:
            self.mark_image = mark_image_contour_and_square

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
            x = self.inverse_transform(x)
            return self.dataset.transform(self.mark_image(x)), y
        else:
            return x, y

    def get_mark_sample_ids(self):
        in_cls = torch.tensor([data[1] in self.cls_to_mark for data in self.dataset])
        torch.manual_seed(27)  # TODO: check best practices for setting seed
        corrupt = torch.rand(len(self.dataset))
        indices = torch.where((corrupt < self.mark_prob) & (in_cls))[0]
        return torch.tensor(indices, dtype=torch.int)
