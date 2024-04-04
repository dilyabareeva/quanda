import os

import torch
from torch.utils.data.dataset import Dataset


class CorruptLabelDataset(Dataset):
    def corrupt_label(self, y):
        ret = y
        while ret == y:
            ret = torch.randint(0, len(self.dataset.classes), (1,))
        return ret

    def __init__(self, dataset, p=0.3):
        super().__init__()
        self.class_labels = dataset.class_labels
        torch.manual_seed(420)  # THIS SHOULD NOT BE CHANGED BETWEEN TRAIN TIME AND TEST TIME
        self.inverse_transform = dataset.inverse_transform
        self.dataset = dataset
        if hasattr(dataset, "class_groups"):
            self.class_groups = dataset.class_groups
        self.classes = dataset.classes
        if os.path.isfile(f"datasets/{dataset.name}_corrupt_ids"):
            self.corrupt_samples = torch.load(f"datasets/{dataset.name}_corrupt_ids")
            self.corrupt_labels = torch.load(f"datasets/{dataset.name}_corrupt_labels")
        else:
            self.corrupt_labels = []
            corrupt = torch.rand(len(dataset))
            self.corrupt_samples = torch.squeeze((corrupt < p).nonzero())
            torch.save(self.corrupt_samples, f"datasets/{dataset.name}_corrupt_ids")
            for i in self.corrupt_samples:
                _, y = self.dataset.__getitem__(i)
                self.corrupt_labels.append(self.corrupt_label(y))
            self.corrupt_labels = torch.tensor(self.corrupt_labels)
            torch.save(self.corrupt_labels, f"datasets/{dataset.name}_corrupt_labels")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y_true = self.dataset[item]
        y = y_true
        if self.dataset.split == "train":
            if item in self.corrupt_samples:
                y = int(self.corrupt_labels[torch.squeeze((self.corrupt_samples == item).nonzero())])
        return x, (y, y_true)
