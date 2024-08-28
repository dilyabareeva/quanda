import glob
import os
import os.path
from typing import Dict, List

import torch
from PIL import Image
from torchvision.datasets import ImageFolder


class CustomDataset(ImageFolder):

    def __init__(self, root: str, classes: List[str], classes_to_idx: Dict[str, int], transform=None, *args, **kwargs):

        self.classes = classes
        self.class_to_idx = classes_to_idx
        super().__init__(root=root, transform=transform, *args, **kwargs)

    def find_classes(self, directory):
        return self.classes, self.class_to_idx


class AnnotatedDataset(torch.utils.data.Dataset):
    def __init__(self, local_path: str, id_dict: dict, annotation: dict, transforms=None):
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
