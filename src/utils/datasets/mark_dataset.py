import os

import torch
from torch.utils.data.dataset import Dataset


class MarkDataset(Dataset):
    def get_mark_sample_ids(self):
        indices = []
        cls = self.cls_to_mark
        prob = self.mark_prob
        for i in range(len(self.dataset)):
            x, y = self.dataset[i]
            if y == cls:
                rnd = torch.rand(1)
                if rnd < prob:
                    indices.append(i)
        return torch.tensor(indices, dtype=torch.int)

    def __init__(self, dataset, p=0.3, cls_to_mark=2, only_train=False):
        super().__init__()
        self.class_labels = dataset.class_labels
        torch.manual_seed(420)  # THIS SHOULD NOT BE CHANGED BETWEEN TRAIN TIME AND TEST TIME
        self.only_train = only_train
        self.dataset = dataset
        self.inverse_transform = dataset.inverse_transform
        self.cls_to_mark = cls_to_mark
        self.mark_prob = p
        if hasattr(dataset, "class_groups"):
            self.class_groups = dataset.class_groups
        self.classes = dataset.classes
        if dataset.split == "train":
            if os.path.isfile(f"datasets/{dataset.name}_mark_ids"):
                self.mark_samples = torch.load(f"datasets/{dataset.name}_mark_ids")
            else:
                self.mark_samples = self.get_mark_sample_ids()
                torch.save(self.mark_samples, f"datasets/{dataset.name}_mark_ids")
        else:
            self.mark_samples = range(len(dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y = self.dataset.__getitem__(item)
        if not self.dataset.split == "train":
            if self.only_train:
                return x, y
            else:
                return self.mark_image(x), y
        else:
            if item in self.mark_samples:
                return self.mark_image(x), y
            else:
                return x, y

    def mark_image_contour(self, x):
        x = self.dataset.inverse_transform(x)
        mask = torch.zeros_like(x[0])
        # for j in range(int(x.shape[-1]/2)):
        #    mask[2*(j):2*(j+1),2*(j):2*(j+1)]=1.
        #    mask[2*j:2*(j+1),-2*(j+1):-2*(j)]=1.
        mask[:2, :] = 1.0
        mask[-2:, :] = 1.0
        mask[:, -2:] = 1.0
        mask[:, :2] = 1.0
        x[0] = torch.ones_like(x[0]) * mask + x[0] * (1 - mask)
        if x.shape[0] > 1:
            x[1:] = torch.zeros_like(x[1:]) * mask + x[1:] * (1 - mask)
        # plt.imshow(x.permute(1,2,0).squeeze())
        # plt.show()

        return self.dataset.transform(x.numpy().transpose(1, 2, 0))

    def mark_image_middle_square(self, x):
        x = self.dataset.inverse_transform(x)
        mask = torch.zeros_like(x[0])
        mid = int(x.shape[-1] / 2)
        mask[(mid - 4) : (mid + 4), (mid - 4) : (mid + 4)] = 1.0
        x[0] = torch.ones_like(x[0]) * mask + x[0] * (1 - mask)
        if x.shape[0] > 1:
            x[1:] = torch.zeros_like(x[1:]) * mask + x[1:] * (1 - mask)
        # plt.imshow(x.permute(1,2,0).squeeze())
        # plt.show()
        return self.dataset.transform(x.numpy().transpose(1, 2, 0))

    def mark_image(self, x):
        x = self.dataset.inverse_transform(x)
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
        # plt.imshow(x.permute(1,2,0).squeeze())
        # plt.show()
        return self.dataset.transform(x.numpy().transpose(1, 2, 0))
