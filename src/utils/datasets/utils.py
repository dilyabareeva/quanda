from torchvision.datasets import CIFAR10, MNIST, FashionMNIST

from src.utils.datasets.corrupt_label_dataset import CorruptLabelDataset
from src.utils.datasets.group_label_dataset import GroupLabelDataset
from src.utils.datasets.mark_dataset import MarkDataset
from src.utils.datasets.reduced_label_dataset import ReduceLabelDataset


def load_datasets(dataset_name, dataset_type, **kwparams):
    ds = None
    evalds = None
    ds_dict = {"MNIST": MNIST, "CIFAR": CIFAR10, "FashionMNIST": FashionMNIST}
    if "only_train" not in kwparams.keys():
        only_train = False
    else:
        only_train = kwparams["only_train"]
    data_root = kwparams["data_root"]
    class_groups = kwparams["class_groups"]
    validation_size = kwparams["validation_size"]
    set = kwparams["image_set"]

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
