import glob
import os
import os.path
import random
from typing import Callable, Dict, List, Optional

import torch
from PIL import Image
from torchvision.datasets import ImageFolder

from quanda.utils.datasets.transformed import (
    LabelFlippingDataset,
    LabelGroupingDataset,
    SampleTransformationDataset,
)


class CustomDataset(ImageFolder):
    def __init__(
        self,
        root: str,
        classes: List[str],
        classes_to_idx: Dict[str, int],
        transform=None,
        *args,
        **kwargs,
    ):
        self.classes = classes
        self.class_to_idx = classes_to_idx
        super().__init__(root=root, transform=transform, *args, **kwargs)

    def find_classes(self, directory):
        return self.classes, self.class_to_idx


class AnnotatedDataset(torch.utils.data.Dataset):
    def __init__(
        self, local_path: str, id_dict: dict, annotation: dict, transforms=None
    ):
        self.filenames = glob.glob(local_path + "/**/*.JPEG")
        self.transform = transforms
        self.id_dict = id_dict
        self.annotation = annotation

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        in_label = self.annotation[os.path.basename(img_path)]
        label = self.id_dict[in_label]
        if self.transform:
            image = self.transform(image)
        return image, label


def special_dataset(
    train_set: torch.utils.data.Dataset,
    n_classes: int,
    new_n_classes: int,
    regular_transforms,
    class_to_group: Dict[int, int],
    shortcut_fn: Callable,
    backdoor_dataset: torch.utils.data.Dataset,
    cat_class: int = 0,
    dog_class: int = 0,
    p_shortcut: float = 1.0,
    p_flipping: float = 1.0,
    pomegranate_class: Optional[int] = None,
    shortcut_transform_indices: List[int] = None,
    flipping_transform_dict: Dict[int, int] = None,
    seed: int = 42,
):
    group_dataset = LabelGroupingDataset(
        dataset=train_set,
        n_classes=n_classes,
        dataset_transform=None,
        class_to_group=class_to_group,
    )

    sc_dataset = SampleTransformationDataset(
        dataset=group_dataset,
        n_classes=new_n_classes,
        dataset_transform=regular_transforms,
        transform_indices=shortcut_transform_indices,
        sample_fn=shortcut_fn,
        cls_idx=pomegranate_class,
        p=p_shortcut,
        seed=seed,
    )

    classes = set(class_to_group.values())

    if flipping_transform_dict is None:
        all_non_transf_idx = [
            i
            for i in range(len(sc_dataset))
            if (
                (i not in sc_dataset.transform_indices)
                and (sc_dataset[i][1] not in [cat_class, dog_class])
            )
        ]
        random_rng = random.Random(seed)
        flip_indices = random_rng.sample(
            all_non_transf_idx, int(p_flipping * len(sc_dataset))
        )
        flipping_transform_dict = {
            i: random_rng.choice(
                list(classes - {sc_dataset[i][1], cat_class, dog_class})
            )
            for i in flip_indices
        }

    flipped = LabelFlippingDataset(
        dataset=sc_dataset,
        n_classes=new_n_classes,
        dataset_transform=None,
        mislabeling_labels=flipping_transform_dict,
        p=p_flipping,
    )

    return torch.utils.data.ConcatDataset([backdoor_dataset, flipped])
