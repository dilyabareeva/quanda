import logging
import os
import sys

import torch.utils

sys.path.append(os.getcwd())

import glob
import math
import subprocess
import warnings
from argparse import ArgumentParser

import lightning as L
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.nn import (
    BCELoss,
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    KLDivLoss,
    MultiMarginLoss,
)
from torch.nn.functional import one_hot
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import (
    AutoAugment,
    AutoAugmentPolicy,
    Compose,
    RandomApply,
    RandomEqualize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    Resize,
)
from tqdm import tqdm

from quanda.benchmarks.resources.sample_transforms import sample_transforms
from quanda.explainers import RandomExplainer
from quanda.explainers.wrappers import (
    TRAK,
    CaptumArnoldi,
    CaptumSimilarity,
    CaptumTracInCPFast,
    RepresenterPoints,
)
from quanda.utils.cache import ExplanationsCache as EC
from quanda.utils.datasets.transformed import (
    LabelFlippingDataset,
    LabelGroupingDataset,
    SampleTransformationDataset,
    TransformedDataset,
)
from quanda.utils.functions import cosine_similarity
from tutorials.utils.datasets import (
    AnnotatedDataset,
    CustomDataset,
    special_dataset,
)

logger = logging.getLogger(__name__)

add_yellow_square = sample_transforms["add_yellow_square"]


class SketchDataset(Dataset):
    def __init__(self, root: str, train: bool, label: int, transform=None, seed=0, *args, **kwargs):
        self.root = root
        self.label = label
        self.transform = transform
        self.train = train

        # find all images in the root directory
        filenames = []
        for dir in os.listdir(root):
            if os.path.isdir(os.path.join(root, dir)):
                filenames += glob.glob(os.path.join(root, dir, "*.JPEG"))
        if os.path.exists(os.path.join(root, "train_indices")):
            train_indices = torch.load(os.path.join(root, "train_indices"))
            test_indices = torch.load(os.path.join(root, "test_indices"))
        else:
            randrank = torch.randperm(len(filenames))
            size = int(len(filenames) / 2)
            train_indices = randrank[:size]
            test_indices = randrank[size:]
            torch.save(train_indices, os.path.join(root, "train_indices"))
            torch.save(test_indices, os.path.join(root, "test_indices"))

        if self.train:
            self.filenames = [filenames[i] for i in train_indices]
        else:
            self.filenames = [filenames[i] for i in test_indices]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.label


def load_scheduler(name, optimizer, epochs):  # include warmup?
    scheduler_dict = {
        "constant": ConstantLR(optimizer=optimizer, last_epoch=-1),
        "step": StepLR(optimizer=optimizer, step_size=epochs // 20, gamma=0.1, last_epoch=epochs),
        "annealing": CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs, last_epoch=epochs
        ),  # make it so that t_max updates to len(train_data) // batch_size (check that this is correct again)
    }
    scheduler = scheduler_dict.get(name, ConstantLR(optimizer=optimizer, last_epoch=-1))
    return scheduler


def load_optimizer(name, model, lr, weight_decay, momentum):  # could add momentum as a variable
    optimizer_dict = {
        "sgd": SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum),
        "adam": Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)),  # No momentum for ADAM
        "rmsprop": RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum),
    }
    optimizer = optimizer_dict.get(name, SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum))
    return optimizer


def load_loss(name):  # add regularisation
    loss_dict = {"cross_entropy": CrossEntropyLoss(), "bce": BCEWithLogitsLoss(reduction="sum"), "hinge": MultiMarginLoss()}
    loss = loss_dict.get(name, CrossEntropyLoss())
    return loss


def load_augmentation(name, dataset_name):
    if name is None or name == "null":
        return lambda x: x
    shapes = {"tiny_imagenet": (64, 64)}
    trans_arr = []
    trans_dict = {
        "crop": RandomApply(
            [
                RandomResizedCrop(
                    size=shapes[dataset_name],
                )
            ],
            p=0.5,
        ),
        "flip": RandomHorizontalFlip(),
        "eq": RandomEqualize(),
        "rotate": RandomApply([RandomRotation(degrees=(0, 180))], p=0.5),
        "cifar": AutoAugment(AutoAugmentPolicy.CIFAR10),
        "imagenet": AutoAugment(AutoAugmentPolicy.IMAGENET),
    }
    for trans in name.split("_"):
        if trans in trans_dict.keys():
            trans_arr.append(trans_dict[trans])
    return Compose(trans_arr)


def evaluate_model(model, device, num_outputs, batch_size, val_set):
    if not torch.cuda.is_available():
        device = "cpu"
    if not len(val_set) > 0:
        return 0.0, 0.0
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    model.eval()
    y_true = torch.empty(0, device=device)
    y_out = torch.empty((0, num_outputs), device=device)

    # for i, (inputs, targets) in enumerate(tqdm(iter(loader), total=len(loader))):
    for i, (inputs, targets) in enumerate(iter(loader)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        y_true = torch.cat((y_true, targets), 0)
        with torch.no_grad():
            logits = model(inputs)
        y_out = torch.cat((y_out, logits), 0)

    y_out = torch.softmax(y_out, dim=1)
    y_pred = torch.argmax(y_out, dim=1)
    model.train()
    return (y_true == y_pred).sum() / y_out.shape[0]


def train_model(
    benchmark_name,
    dataset_name,
    dataset_type,
    dataset_path,
    metadata_path,
    download,
    num_groups,
    device,
    batch_size,
    models_dir,
    output_dir,
):
    torch.set_float32_matmul_precision("medium")
    seed = 4242
    torch.manual_seed(seed)
    # Downloading the datasets and checkpoints

    if download:
        os.makedirs(metadata_path, exist_ok=True)
        os.makedirs(dataset_path, exist_ok=True)

        if not os.path.join(dataset_path, "tiny-imagenet-200.zip"):
            subprocess.run(["wget", "-qP", dataset_path, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])
        # if not os.path.join(metadata_path, "sketch.zip"):
        #     subprocess.run(
        #         ["wget", "-qP", metadata_path, "https://datacloud.hhi.fraunhofer.de/s/FpPWkzPmM3s9ZqF/download/sketch.zip"]
        #     )
    if not os.path.exists(os.path.join(dataset_path, "tiny-imagenet-200")):
        subprocess.run(["unzip", "-qq", os.path.join(dataset_path, "tiny-imagenet-200.zip"), "-d", dataset_path])
    if not os.path.exists(os.path.join(metadata_path, "sketch")):
        subprocess.run(["unzip", "-qq", os.path.join(metadata_path, "sketch.zip"), "-d", metadata_path])

    # Dataset Construction

    # Define transformations
    regular_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    resize_regular_transform = transforms.Compose([Resize(size=(64, 64)), regular_transforms])

    denormalize = transforms.Compose(
        [transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])]
        + [transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])]
    )

    # Load the TinyImageNet dataset
    dataset_path = os.path.join(dataset_path, "tiny-imagenet-200")
    with open(os.path.join(dataset_path, "wnids.txt"), "r") as f:
        id_dict = {line.strip(): i for i, line in enumerate(f)}

    with open(os.path.join(dataset_path, "val/val_annotations.txt"), "r") as f:
        val_annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}

    normalized_transform = regular_transforms
    train_set = CustomDataset(
        os.path.join(dataset_path, "train"),
        classes=list(id_dict.keys()),
        classes_to_idx=id_dict,
        transform=None if dataset_type in ["mislabeled", "shortcut"] else normalized_transform,
    )

    holdout_set = AnnotatedDataset(
        local_path=os.path.join(dataset_path, "val"),
        transforms=normalized_transform,
        id_dict=id_dict,
        annotation=val_annotations,
    )
    if os.path.exists(os.path.join(metadata_path, "validation_indices")):
        val_indices = torch.load(os.path.join(metadata_path, "validation_indices"))
        test_indices = torch.load(os.path.join(metadata_path, "test_indices"))
    else:
        all_indices = torch.randperm(len(holdout_set))
        val_indices = all_indices[:3000]
        test_indices = all_indices[3000:]
        torch.save(val_indices, os.path.join(metadata_path, "validation_indices"))
        torch.save(test_indices, os.path.join(metadata_path, "test_indices"))
    val_set = Subset(holdout_set, val_indices)
    test_set = Subset(holdout_set, test_indices)

    wn_ids = {}
    with open(os.path.join(dataset_path, "words.txt"), "r") as f:
        lines = f.readlines()
        for l in lines:
            l = l.replace("\n", "")
            l = l.split("\t")
            if l[0] in train_set.classes:
                wn_ids[l[1]] = l[0]

    pg_id = train_set.class_to_idx[wn_ids["pomegranate"]]
    bb_id = train_set.class_to_idx[wn_ids["basketball"]]

    # Initialize model
    model = resnet18()

    num_outputs = 200 if num_groups is None else num_groups
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_outputs)
    model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    model.maxpool = torch.nn.Sequential()
    model.to(device=device)

    # conv_list = [mod for name, mod in model.named_modules() if "conv" in name]

    # def hook(mod, inp, out):
    #     print(f"{out.shape} - kernel {mod.kernel_size} - stride {mod.stride} - padding {mod.padding}")

    # for m in conv_list:
    #     m.register_forward_hook(hook)

    if dataset_type == "shortcut":
        if not os.path.exists(os.path.join(metadata_path, "shortcut_indices")):
            raise ValueError("Shortcut indices not found")
        else:
            shortcut_indices = torch.load(os.path.join(metadata_path, "shortcut_indices"))
            train_set = SampleTransformationDataset(
                dataset=train_set,
                n_classes=200,
                sample_fn=add_yellow_square,
                dataset_transform=regular_transforms,
                transform_indices=shortcut_indices,
                seed=seed,
            )
        # plt.imshow(denormalize(train_set[train_set.transform_indices[0]][0]).permute(1, 2, 0))
        # plt.show(block=True)
    elif dataset_type == "mislabeled":
        if not os.path.exists(os.path.join(metadata_path, "mislabeling_indices")):
            raise ValueError("Mislabeling indices not found")
        else:
            mislabeling_indices = torch.load(os.path.join(metadata_path, "mislabeling_indices"))
            mislabeling_labels = torch.load(os.path.join(metadata_path, "mislabeling_labels"))
            mislabeling_labels = {
                mislabeling_indices[i]: int(mislabeling_labels[i]) for i in range(mislabeling_labels.shape[0])
            }
            train_set = LabelFlippingDataset(
                dataset=train_set,
                n_classes=200,
                dataset_transform=regular_transforms,
                transform_indices=mislabeling_indices,
                mislabeling_labels=mislabeling_labels,
                seed=seed,
            )
        # for i in range(10):
        # plt.imshow(train_set[mislabeling_indices[i]][0].permute(1, 2, 0))
        # plt.show(block=True)
        # print(train_set[mislabeling_indices[i]][1]!=train_set.dataset.targets[mislabeling_indices[i]])
    elif dataset_type == "mixed":
        adversarial_dataset = SketchDataset(
            root=os.path.join(metadata_path, "sketch"), label=bb_id, transform=resize_regular_transform, train=True
        )
        adversarial_indices = [1] * len(adversarial_dataset) + [0] * len(train_set)
        train_set = torch.utils.data.ConcatDataset([adversarial_dataset, train_set])

    bench_state = {}

    ckpt_names = []
    ckpt_binary = []
    file_list = [f for f in os.listdir(models_dir) if f.endswith(".ckpt")]
    file_list = sorted(file_list)
    for file in file_list:
        ckpt_names.append(file)
        model_state_dict = torch.load(os.path.join(models_dir, file), map_location=device)
        ckpt_binary.append(model_state_dict["model_state"])

    bench_state["checkpoints"] = ckpt_names
    bench_state["checkpoints_binary"] = ckpt_binary
    bench_state["dataset_str"] = "zh-plus/tiny-imagenet"
    bench_state["use_predictions"] = True
    bench_state["n_classses"] = num_outputs
    test_idx_selection = torch.randperm(len(test_set))[:128]
    bench_state["eval_test_indices"] = [i.item() for i in test_indices[test_idx_selection]]
    bench_state["dataset_transform"] = "tiny_imagenet_transforms"
    bench_state["pl_module"] = "TinyImagenetModel"

    if benchmark_name == "mislabeling_detection":
        bench_state["mislabeling_labels"] = mislabeling_labels
    elif benchmark_name == "shortcut_detection":
        bench_state["shortcut_cls"] = pg_id
        bench_state["shortcut_indices"] = shortcut_indices
        bench_state["sample_fn"] = "add_yellow_square"
    elif benchmark_name == "mixed_detection":
        bench_state["adversarial_label"] = bb_id
        bench_state["adversarial_transform"] = "tiny_imageent_adversarial_transform"
        bench_state["adversarial_dir_url"] = None
    torch.save(bench_state, os.path.join(output_dir, f"tiny_imagenet_{benchmark_name}.pth"))


if __name__ == "__main__":
    parser = ArgumentParser()

    # Define argument for method with choices
    parser.add_argument("--benchmark_name", required=True, default="class_detection", type=str, help="Name of the dataset")
    parser.add_argument("--dataset_name", required=True, default="tiny_imagenet", type=str, help="Name of the dataset")
    parser.add_argument("--dataset_type", required=True, choices=["vanilla", "mislabeled", "shortcut", "mixed", "subclass"])

    # Define other required arguments
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to Tiny ImageNet dataset")
    parser.add_argument("--models_dir", required=False, type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--metadata_path", required=False, default=None)
    parser.add_argument("--num_groups", required=False, type=int, default=None)
    parser.add_argument("--batch_size", required=True, type=int, default=64)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--download", action="store_true", help="Download the datasets and checkpoints")
    parser.add_argument("--device", required=True, type=str, help="Device to run the model on", choices=["cpu", "cuda"])

    args = parser.parse_args()

    assert not (args.num_groups is None and args.dataset_type == "subclass")
    assert not (args.num_groups is not None and args.dataset_type != "subclass")

    # Call the function with parsed arguments
    train_model(
        args.benchmark_name,
        args.dataset_name,
        args.dataset_type,
        args.dataset_path,
        args.metadata_path,
        args.download,
        args.num_groups,
        args.device,
        args.batch_size,
        args.models_dir,
        args.output_dir,
    )
