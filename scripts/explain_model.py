import logging
import os
import sys

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
)
from tqdm import tqdm

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
from tutorials.utils.modules import LitModel

logger = logging.getLogger(__name__)


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


# def load_scheduler(name, optimizer, epochs):  # include warmup?
#     scheduler_dict = {
#         "constant": ConstantLR(optimizer=optimizer, last_epoch=-1),
#         # "step": StepLR(optimizer=optimizer, step_size=epochs // 20, gamma=0.1, last_epoch=epochs),
#         "annealing": CosineAnnealingLR(
#             optimizer=optimizer, T_max=epochs, last_epoch=epochs
#         ),  # make it so that t_max updates to len(train_data) // batch_size (check that this is correct again)
#     }
#     scheduler = scheduler_dict.get(name, ConstantLR(optimizer=optimizer, last_epoch=-1))
#     return scheduler


# def load_optimizer(name, model, lr, weight_decay, momentum):  # could add momentum as a variable
#     optimizer_dict = {
#         "sgd": SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum),
#         "adam": Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(momentum, 0.999)),  # No momentum for ADAM
#         "rmsprop": RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum),
#     }
#     optimizer = optimizer_dict.get(name, SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum))
#     if name == "adam":
#         warnings.warn("For Adam, the given momentum value is used for beta_1.")
#     return optimizer


def load_loss(name):  # add regularisation
    loss_dict = {"cross_entropy": CrossEntropyLoss(), "bce": BCEWithLogitsLoss(reduction="sum"), "hinge": MultiMarginLoss()}
    loss = loss_dict.get(name, CrossEntropyLoss())
    return loss


# def load_augmentation(name, dataset_name):
#     if name is None:
#         return lambda x: x
#     shapes = {"MNIST": (28, 28), "CIFAR": (32, 32), "AWA": (224, 224)}
#     trans_arr = []
#     trans_dict = {
#         "crop": RandomApply(
#             [
#                 RandomResizedCrop(
#                     size=shapes[dataset_name],
#                 )
#             ],
#             p=0.5,
#         ),
#         "flip": RandomHorizontalFlip(),
#         "eq": RandomEqualize(),
#         "rotate": RandomApply([RandomRotation(degrees=(0, 180))], p=0.5),
#         "cifar": AutoAugment(AutoAugmentPolicy.CIFAR10),
#         "imagenet": AutoAugment(AutoAugmentPolicy.IMAGENET),
#     }
#     for trans in name.split("_"):
#         if trans in trans_dict.keys():
#             trans_arr.append(trans_dict[trans])
#     return Compose(trans_arr)


def load_state_dict(module: L.LightningModule, path: str) -> int:
    checkpoints = torch.load(path, map_location=torch.device("cuda:0"))
    module.model.load_state_dict(checkpoints["model_state_dict"])
    module.eval()
    return module.lr


def load_explainer(name, train_dataset, model, cache_dir, ckpt_dir, device, batch_size):
    explainers = {
        "similarity": CaptumSimilarity(
            model=model,
            model_id="0",
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            layers="avgpool",
            similarity_metric=cosine_similarity,
            device=device,
            batch_size=batch_size,
            load_from_disk=False,
        ),
        "representer": RepresenterPoints(
            model=model,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            features_layer="avgpool",
            classifier_layer="fc",
            batch_size=batch_size,
            features_postprocess=None,
            model_id="demo",
            load_from_disk=False,
            show_progress=False,
        ),
        "tracin": CaptumTracInCPFast(
            model=model,
            train_dataset=train_dataset,
            checkpoints=[i for i in list(os.listdir(ckpt_dir)) if os.path.isfile(i)],
            checkpoints_load_func=load_state_dict,
            loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
            final_fc_layer=model.fc,
            device=device,
            batch_size=batch_size * 4,
        ),
        "arnoldi": CaptumArnoldi(
            model=model,
            train_dataset=train_dataset,
            hessian_dataset=Subset(train_dataset, torch.randperm(len(train_dataset))[:5000]),
            checkpoint=model.state_dict(),
            loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
            checkpoints_load_func=load_state_dict,
            projection_dim=100,
            arnoldi_dim=200,
            batch_size=batch_size * 4,
            layers=["model.fc"],  # only the last layer
            device=device,
        ),
        "trak": TRAK(
            model=model,
            model_id="test_model",
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            projector="cuda",
            proj_dim=4096,
            batch_size=batch_size,
            load_from_disk=False,
        ),
        "random": RandomExplainer(
            model=model,
            train_dataset=train_dataset,
            seed=4242,
        ),
    }
    return explainers[name]


def add_yellow_square(img):
    square_size = (15, 15)  # Size of the square
    yellow_square = Image.new("RGB", square_size, (255, 255, 0))  # Create a yellow square
    img.paste(yellow_square, (10, 10))  # Paste it onto the image at the specified position
    return img


# def evaluate_model(model, device, num_outputs, batch_size, val_set):
#     if not torch.cuda.is_available():
#         device = "cpu"
#     if not len(val_set) > 0:
#         return 0.0, 0.0
#     loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

#     model.eval()
#     y_true = torch.empty(0, device=device)
#     y_out = torch.empty((0, num_outputs), device=device)

#     for i, (inputs, targets) in enumerate(tqdm(iter(loader), total=len(loader))):
#         inputs = inputs.to(device)
#         targets = targets.to(device)
#         y_true = torch.cat((y_true, targets), 0)
#         with torch.no_grad():
#             logits = model(inputs)
#         y_out = torch.cat((y_out, logits), 0)

#     y_out = torch.softmax(y_out, dim=1)
#     y_pred = torch.argmax(y_out, dim=1)
#     model.train()
#     return (y_true == y_pred).sum() / y_out.shape[0]


def explain_model(
    explainer,
    dataset_type,
    tiny_imgnet_path,
    metadata_path,
    cache_dir_base,
    ckpt_dir_base,
    output_dir,
    download,
    device,
    lr,
    batch_size,
    loss,
    model_path,
):
    torch.set_float32_matmul_precision("medium")
    seed = 4242
    torch.manual_seed(seed)
    # Downloading the datasets and checkpoints
    os.makedirs(output_dir, exist_ok=True)

    if download:
        os.makedirs(metadata_path, exist_ok=True)
        os.makedirs(tiny_imgnet_path, exist_ok=True)

        if not os.path.join(tiny_imgnet_path, "tiny-imagenet-200.zip"):
            subprocess.run(["wget", "-qP", tiny_imgnet_path, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])
        # if not os.path.join(metadata_path, "sketch.zip"):
        #     subprocess.run(
        #         ["wget", "-qP", metadata_path, "https://datacloud.hhi.fraunhofer.de/s/FpPWkzPmM3s9ZqF/download/sketch.zip"]
        #     )
    if not os.path.exists(os.path.join(tiny_imgnet_path, "tiny-imagenet-200")):
        subprocess.run(["unzip", "-qq", os.path.join(tiny_imgnet_path, "tiny-imagenet-200.zip"), "-d", tiny_imgnet_path])
    if not os.path.exists(os.path.join(metadata_path, "sketch")):
        subprocess.run(["unzip", "-qq", os.path.join(metadata_path, "sketch.zip"), "-d", metadata_path])

    # Dataset Construction

    # Define transformations
    regular_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    denormalize = transforms.Compose(
        [transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])]
        + [transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])]
    )

    save_id_base = f"{dataset_type}_{explainer}"
    # Load the TinyImageNet dataset
    tiny_imgnet_path = os.path.join(tiny_imgnet_path, "tiny-imagenet-200")
    with open(os.path.join(tiny_imgnet_path, "wnids.txt"), "r") as f:
        id_dict = {line.strip(): i for i, line in enumerate(f)}

    with open(os.path.join(tiny_imgnet_path, "val/val_annotations.txt"), "r") as f:
        val_annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}

    train_set = CustomDataset(
        os.path.join(tiny_imgnet_path, "train"),
        classes=list(id_dict.keys()),
        classes_to_idx=id_dict,
        transforms=None if dataset_type in ["shortcut", "mislabeled"] else regular_transform,
    )

    holdout_set = AnnotatedDataset(
        local_path=os.path.join(tiny_imgnet_path, "val"),
        transforms=None if dataset_type == "shortcut" else regular_transform,
        id_dict=id_dict,
        annotation=val_annotations,
    )

    val_indices = torch.load(os.path.join(metadata_path, "validation_indices"))
    test_indices = torch.load(os.path.join(metadata_path, "test_indices"))
    test_set = Subset(holdout_set, test_indices)

    wn_ids = {}
    with open(os.path.join(tiny_imgnet_path, "words.txt"), "r") as f:
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
    ## TODO load model weights

    num_outputs = 200
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_outputs)
    model.to(device=device)

    loss = load_loss(loss)

    if dataset_type == "shortcut":
        shortcut_indices = torch.load(os.path.join(metadata_path, "shortcut_indices"))
        train_set = SampleTransformationDataset(
            dataset=train_set,
            n_classes=200,
            sample_fn=add_yellow_square,
            dataset_transform=regular_transform,
            transform_indices=shortcut_indices,
            seed=seed,
        )

        test_set = SampleTransformationDataset(
            dataset=test_set,
            n_classes=200,
            sample_fn=add_yellow_square,
            dataset_transform=regular_transform,
            transform_indices=shortcut_indices,
            seed=seed,
        )

        # plt.imshow(denormalize(train_set[train_set.transform_indices[0]][0]).permute(1, 2, 0))
        # plt.show(block=True)
    elif dataset_type == "mislabeled":
        mislabeling_indices = torch.load(os.path.join(metadata_path, "mislabeling_indices"))
        mislabeling_labels = torch.load(os.path.join(metadata_path, "mislabeling_labels"))
        train_set = LabelFlippingDataset(
            dataset=train_set,
            n_classes=200,
            dataset_transform=regular_transform,
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
            root=os.path.join(metadata_path, "sketch"), label=bb_id, transform=regular_transform, train=True
        )
        adversarial_indices = [1] * len(adversarial_dataset) + [0] * len(train_set)
        train_set = torch.utils.data.ConcatDataset([adversarial_dataset, train_set])
        test_set = SketchDataset(
            root=os.path.join(metadata_path, "sketch"), label=bb_id, transform=regular_transform, train=False
        )
        # print(train_set[0][1])
        # plt.imshow(denormalize(train_set[0][0]).permute(1, 2, 0))
        # plt.show(block=True)
        # print(train_set[10][1])
        # plt.imshow(denormalize(train_set[10][0]).permute(1, 2, 0))
        # plt.show(block=True)
        # print(train_set[len(adversarial_dataset)][1])
        # plt.imshow(denormalize(train_set[len(adversarial_dataset)-1][0]).permute(1, 2, 0))
        # plt.show(block=True)
        # print(train_set[len(adversarial_dataset)+1][1])
        # plt.imshow(denormalize(train_set[len(adversarial_dataset)][0]).permute(1, 2, 0))
        # plt.show(block=True)
        # exit()
    #    elif dataset_type == "subclass":
    #        pass

    model.eval()
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # TODO: initialize explainer
    cache_dir = os.path.join(cache_dir_base, dataset_type, explainer)
    ckpt_dir = os.path.join(ckpt_dir_base, dataset_type, explainer)
    os.makedirs(cache_dir, exist_ok=True)
    explainer = load_explainer(explainer, train_set, model, cache_dir, ckpt_dir, device, 32)

    # TODO: generate test explanations
    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    explanations = torch.tensor(0, len(train_set))
    for x, y in iter(loader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=-1)
        explanations = torch.concat((explanations, explainer.explain()))
        torch.save()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Define argument for method with choices
    parser.add_argument("--dataset_type", required=True, choices=["vanilla", "mislabeled", "shortcut", "mixed", "subclass"])

    # Define other required arguments
    parser.add_argument("--tiny_imgnet_path", required=True, type=str, help="Path to Tiny ImageNet dataset")
    parser.add_argument("--ckpt_dir", required=True, type=str, help="Path to Tiny ImageNet dataset")
    parser.add_argument("--model_path", required=False, type=str, default=None, help="Path to Tiny ImageNet dataset")
    parser.add_argument("--metadata_path", required=False, default=None)
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save outputs")
    parser.add_argument("--num_groups", required=False, type=int, default=None)
    parser.add_argument("--epochs", required=True, type=int, default=100)
    parser.add_argument("--save_each", required=True, type=int, default=10)
    parser.add_argument("--batch_size", required=True, type=int, default=64)
    parser.add_argument("--download", action="store_true", help="Download the datasets and checkpoints")
    parser.add_argument("--device", required=True, type=str, help="Device to run the model on", choices=["cpu", "cuda"])
    parser.add_argument("--loss", required=False, type=str, default="cross_entropy")

    args = parser.parse_args()

    assert not (args.num_groups is None and args.dataset_type == "subclass")
    assert not (args.num_groups is not None and args.dataset_type != "subclass")

    # Call the function with parsed arguments
    explain_model(
        args.dataset_type,
        args.tiny_imgnet_path,
        args.metadata_path,
        args.ckpt_dir,
        args.output_dir,
        args.download,
        args.class_groups,
        args.device,
        args.batch_size,
        args.loss,
        args.augmentation,
        args.model_path,
    )
