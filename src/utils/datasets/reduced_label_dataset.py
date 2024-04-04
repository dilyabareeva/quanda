from torch.utils.data.dataset import Dataset


class ReduceLabelDataset(Dataset):
    def __init__(self, dataset, first=True):
        super().__init__()
        self.dataset = dataset
        if hasattr(dataset, "class_groups"):
            self.class_groups = dataset.class_groups
        self.classes = dataset.classes
        self.first = first

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, (y, c) = self.dataset[item]
        if self.first:
            return x, y
        else:
            return x, c
