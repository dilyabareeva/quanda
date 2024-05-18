from torch.utils.data.dataset import Dataset

CLASS_GROUP_BY = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
TRANSFORM_DICT = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 4}


class LabelTransformDataset(Dataset):
    """
    Meant to replace: GroupLabelDataset, CorruptLabelDataset

    """

    def __init__(self, dataset, transform_dict=None):
        self.dataset = dataset
        self.inverse_transform = dataset.inverse_transform
        if transform_dict is None:
            transform_dict = TRANSFORM_DICT
        self.transform_dict = transform_dict
        self.inv_transform_dict = self.invert_labels_dict(transform_dict)
        self.class_labels = list(self.inv_transform_dict.keys())

    def __getitem__(self, index):
        x, y = self.dataset[index]
        g = self.transform_dict[y]
        return x, (g, y)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def invert_labels_dict(labels_dict):
        return {v: [k for k in labels_dict if labels_dict[k] == v] for v in set(labels_dict.values())}
