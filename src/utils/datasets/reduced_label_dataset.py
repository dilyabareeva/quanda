from torch.utils.data.dataset import Dataset


class ReduceLabelDataset(Dataset):
    def __init__(self, dataset, classes, class_groups, first=True):
        super().__init__()
        self.dataset = dataset
        self.class_groups = class_groups
        self.classes = classes
        self.first = first

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, (y, c) = self.dataset[item]
        if self.first:
            return x, y
        else:
            return x, c
