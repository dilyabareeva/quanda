from torch.utils.data.dataset import Dataset

CLASS_GROUP_BY = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


class GroupLabelDataset(Dataset):
    def __init__(self, dataset, class_groups=None):
        self.dataset = dataset
        self.class_labels = [i for i in range(len(class_groups))]
        self.inverse_transform = dataset.inverse_transform
        if class_groups is None:
            class_groups = CLASS_GROUP_BY
        self.class_groups = class_groups
        self.inverted_class_groups = self.invert_class_groups(class_groups)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        g = self.inverted_class_groups[y]
        return x, (g, y)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def invert_class_groups(groups):
        inverted_class_groups = {}
        for g, group in enumerate(groups):
            intersection = inverted_class_groups.keys() & group
            if len(intersection) > 0:
                raise ValueError("Class indices %s are present in multiple groups." % (str(intersection)))
            inverted_class_groups.update({cls: g for cls in group})
        return inverted_class_groups
