from torch.utils.data.dataset import Dataset


class GroupLabelDataset(Dataset):
    class_group_by2 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def __init__(self, dataset, class_groups=None):
        self.dataset = dataset
        self.class_labels = [i for i in range(len(class_groups))]
        self.inverse_transform = dataset.inverse_transform
        if class_groups is None:
            class_groups = GroupLabelDataset.class_group_by2
        self.classes = class_groups
        GroupLabelDataset.check_class_groups(class_groups)

    def __getitem__(self, item):
        x, y = self.dataset.__getitem__(item)
        g = -1
        for i, group in enumerate(self.classes):
            if y in group:
                g = i
                break
        return x, (g, y)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def check_class_groups(groups):
        vals = [[] for _ in range(10)]
        for g, group in enumerate(groups):
            for i in group:
                vals[i].append(g)
        for v in vals:
            assert len(v) == 1  # Check that this is the first time i is encountered
