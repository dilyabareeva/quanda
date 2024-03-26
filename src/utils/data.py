from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from datasets.MNIST import MNIST, FashionMNIST
from datasets.CIFAR import CIFAR
import matplotlib.pyplot as plt
import os


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
        x, (y, c) = self.dataset.__getitem__(item)
        if self.first:
            return x, y
        else:
            return x, c


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
        if os.path.isfile(f'datasets/{dataset.name}_corrupt_ids'):
            self.corrupt_samples = torch.load(f'datasets/{dataset.name}_corrupt_ids')
            self.corrupt_labels = torch.load(f'datasets/{dataset.name}_corrupt_labels')
        else:
            self.corrupt_labels = []
            corrupt = torch.rand(len(dataset))
            self.corrupt_samples = torch.squeeze((corrupt < p).nonzero())
            torch.save(self.corrupt_samples, f'datasets/{dataset.name}_corrupt_ids')
            for i in self.corrupt_samples:
                _, y = self.dataset.__getitem__(i)
                self.corrupt_labels.append(self.corrupt_label(y))
            self.corrupt_labels = torch.tensor(self.corrupt_labels)
            torch.save(self.corrupt_labels, f"datasets/{dataset.name}_corrupt_labels")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        x, y_true = self.dataset.__getitem__(item)
        y = y_true
        if self.dataset.split == "train":
            if item in self.corrupt_samples:
                y = int(self.corrupt_labels[torch.squeeze((self.corrupt_samples == item).nonzero())])
        return x, (y, y_true)


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
            if os.path.isfile(f'datasets/{dataset.name}_mark_ids'):
                self.mark_samples = torch.load(f'datasets/{dataset.name}_mark_ids')
            else:
                self.mark_samples = self.get_mark_sample_ids()
                torch.save(self.mark_samples, f'datasets/{dataset.name}_mark_ids')
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
        mask[:2, :] = 1.
        mask[-2:, :] = 1.
        mask[:, -2:] = 1.
        mask[:, :2] = 1.
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
        mask[mid - 4:mid + 4, mid - 4:mid + 4] = 1.
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
        mask[mid - 3:mid + 3, mid - 3:mid + 3] = 1.
        mask[:2, :] = 1.
        mask[-2:, :] = 1.
        mask[:, -2:] = 1.
        mask[:, :2] = 1.
        x[0] = torch.ones_like(x[0]) * mask + x[0] * (1 - mask)
        if x.shape[0] > 1:
            x[1:] = torch.zeros_like(x[1:]) * mask + x[1:] * (1 - mask)
        # plt.imshow(x.permute(1,2,0).squeeze())
        # plt.show()
        return self.dataset.transform(x.numpy().transpose(1, 2, 0))


class GroupLabelDataset(Dataset):
    class_group_by2 = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    @staticmethod
    def check_class_groups(groups):
        vals = [[] for _ in range(10)]
        for g, group in enumerate(groups):
            for i in group:
                vals[i].append(g)
        for v in vals:
            assert (len(v) == 1)  # Check that this is the first time i is encountered

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


class FeatureDataset(Dataset):
    def __init__(self, model, dataset, device, file=None):
        self.model = model
        self.device = device
        self.samples = torch.empty(size=(0, model.classifier.in_features), device=self.device)
        self.labels = torch.empty(size=(0,), device=self.device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        super().__init__()
        if file is not None:
            self.load_from_file(file)
        else:
            for x, y in tqdm(iter(loader)):
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    x = model.features(x)
                    self.samples = torch.cat((self.samples, x), 0)
                    self.labels = torch.cat((self.labels, y), 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item], self.labels[item]

    def load_from_file(self, file):
        if ".csv" in file:
            from utils.csv_io import read_matrix
            mat = read_matrix(file_name=file)
            self.samples = torch.tensor(mat[:, :-1], dtype=torch.float, device=self.device)
            self.labels = torch.tensor(mat[:, -1], dtype=torch.int, device=self.device)
        else:
            self.samples = torch.load(os.path.join(file, "samples_tensor"), map_location=self.device)
            self.labels = torch.load(os.path.join(file, "labels_tensor"), map_location=self.device)


class RestrictedDataset(Dataset):
    def __init__(self, dataset, indices, return_indices=False):
        self.dataset = dataset
        self.indices = indices
        self.return_indices = return_indices
        if hasattr(dataset, "name"):
            self.name = dataset.name
        else:
            self.name = dataset.dataset.name

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        d = self.dataset[self.indices[item]]
        if self.return_indices:
            return d, self.indices[item]
        return d


def load_datasets(dataset_name, dataset_type, **kwparams):
    ds = None
    evalds = None
    ds_dict = {'MNIST': MNIST, 'CIFAR': CIFAR, 'FashionMNIST': FashionMNIST}
    if "only_train" not in kwparams.keys():
        only_train = False
    else:
        only_train = kwparams['only_train']
    data_root = kwparams['data_root']
    class_groups = kwparams['class_groups']
    validation_size = kwparams['validation_size']
    set = kwparams['image_set']

    if dataset_name in ds_dict.keys():
        dscls = ds_dict[dataset_name]
        ds = dscls(root=data_root, split="train", validation_size=validation_size)
        evalds = dscls(root=data_root, split=set, validation_size=validation_size)
    else:
        raise NameError(f"Unresolved dataset name : {dataset_name}.")
    if dataset_type == "group":
        ds = GroupLabelDataset(ds, class_groups=class_groups)
        evalds = GroupLabelDataset(evalds, class_groups=class_groups)
    elif dataset_type == "corrupt":
        ds = CorruptLabelDataset(ds)
        evalds = CorruptLabelDataset(evalds)
    elif dataset_type == "mark":
        ds = MarkDataset(ds, only_train=only_train)
        evalds = MarkDataset(evalds, only_train=only_train)
    assert ds is not None and evalds is not None
    return ds, evalds


def load_datasets_reduced(dataset_name, dataset_type, kwparams):
    ds, evalds = load_datasets(dataset_name, dataset_type, **kwparams)
    if dataset_type in ["group", "corrupt"]:
        ds = ReduceLabelDataset(ds)
        evalds = ReduceLabelDataset(evalds)
    return ds, evalds