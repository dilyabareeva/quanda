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
    Resize,
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
    if name == "adam":
        warnings.warn("For Adam, the given momentum value is used for beta_1.")
    return optimizer


def load_loss(name):  # add regularisation
    loss_dict = {"cross_entropy": CrossEntropyLoss(), "bce": BCEWithLogitsLoss(reduction="sum"), "hinge": MultiMarginLoss()}
    loss = loss_dict.get(name, CrossEntropyLoss())
    return loss


def load_augmentation(name, dataset_name):
    if name is None:
        return lambda x: x
    shapes = {"MNIST": (28, 28), "CIFAR": (32, 32), "AWA": (224, 224)}
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


def add_yellow_square(img):
    square_size = (15, 15)  # Size of the square
    yellow_square = Image.new("RGB", square_size, (255, 255, 0))  # Create a yellow square
    img.paste(yellow_square, (10, 10))  # Paste it onto the image at the specified position
    return img


def evaluate_model(model, device, num_outputs, batch_size, val_set):
    if not torch.cuda.is_available():
        device = "cpu"
    if not len(val_set) > 0:
        return 0.0, 0.0
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    model.eval()
    y_true = torch.empty(0, device=device)
    y_out = torch.empty((0, num_outputs), device=device)

    #for i, (inputs, targets) in enumerate(tqdm(iter(loader), total=len(loader))):
    for i, (inputs, targets) in enumerate(iter(loader), total=len(loader)):
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
    dataset_type,
    tiny_imgnet_path,
    metadata_path,
    output_dir,
    download,
    num_groups,
    device,
    pretrained,
    epochs,
    lr,
    batch_size,
    save_each,
    optimizer,
    weight_decay,
    momentum,
    scheduler,
    loss,
    augmentation,
    model_path,
    base_epoch,
):
    torch.set_float32_matmul_precision("medium")
    seed = 4242
    torch.manual_seed(seed)
    # Downloading the datasets and checkpoints
    os.makedirs(output_dir, exist_ok=True)
    save_id_base = f"{dataset_type}_{lr}_{scheduler}_{optimizer}{f'_aug' if augmentation is not None else ''}"
    print(save_id_base)
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
    regular_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    resize_regular_transform = transforms.Compose([Resize(size=(64, 64)), regular_transforms])

    denormalize = transforms.Compose(
        [transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])]
        + [transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])]
    )
    if augmentation == "":
        augmentation = None
    if augmentation is not None:
        augmentation = load_augmentation(augmentation)

    # Load the TinyImageNet dataset
    tiny_imgnet_path = os.path.join(tiny_imgnet_path, "tiny-imagenet-200")
    with open(os.path.join(tiny_imgnet_path, "wnids.txt"), "r") as f:
        id_dict = {line.strip(): i for i, line in enumerate(f)}

    with open(os.path.join(tiny_imgnet_path, "val/val_annotations.txt"), "r") as f:
        val_annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}

    normalized_transform = (
        transforms.Compose([augmentation, regular_transforms]) if augmentation is not None else regular_transforms
    )
    train_set = CustomDataset(
        os.path.join(tiny_imgnet_path, "train"),
        classes=list(id_dict.keys()),
        classes_to_idx=id_dict,
        transform=augmentation if dataset_type in ["mislabeled", "shortcut"] else normalized_transform,
    )

    holdout_set = AnnotatedDataset(
        local_path=os.path.join(tiny_imgnet_path, "val"),
        transforms=regular_transforms,
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
    val_set = Subset(holdout_set, val_indices)

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
    if pretrained:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18()

    num_outputs = 200 if num_groups is None else num_groups
    model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_outputs)
    model.to(device=device)

    tensorboarddir = f"{dataset_type}_{lr}_{scheduler}_{optimizer}{f'_aug' if augmentation is not None else ''}"
    tensorboarddir = os.path.join(output_dir, tensorboarddir)
    writer = SummaryWriter(tensorboarddir)
    loss = load_loss(loss)
    optimizer = load_optimizer(optimizer, model, lr, weight_decay, momentum)
    scheduler = load_scheduler(scheduler, optimizer, epochs)

    if dataset_type == "shortcut":
        if not os.path.exists(os.path.join(metadata_path, "shortcut_indices")):
            train_set = SampleTransformationDataset(
                dataset=train_set,
                n_classes=200,
                sample_fn=add_yellow_square,
                dataset_transform=regular_transforms,
                cls_idx=pg_id,
                p=0.2,
                seed=seed,
            )
            torch.save(train_set.transform_indices, os.path.join(output_dir, "shortcut_indices"))
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
            train_set = LabelFlippingDataset(
                dataset=train_set, n_classes=200, dataset_transform=regular_transforms, p=0.3, seed=seed
            )
            mislabeling_indices = train_set.transform_indices
            torch.save(mislabeling_indices, os.path.join(output_dir, "mislabeling_indices"))
            mislabeling_labels = torch.tensor([train_set.mislabeling_labels[i] for i in mislabeling_indices])
            torch.save(mislabeling_labels, os.path.join(output_dir, "mislabeling_labels"))
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

    learning_rates = []
    train_losses = []
    validation_losses = []
    validation_epochs = []
    val_acc = []
    train_acc = []

    saved_files = []

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        if checkpoint.get("model_state", None) != None:
            model.load_state_dict(checkpoint["model_state"])
        if checkpoint.get("optimizer_state", None) != None:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if checkpoint.get("scheduler_state", None) != None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("train_losses", None) != None:
            train_losses = checkpoint.get("train_losses", None)
        if checkpoint.get("validation_losses", None) != None:
            validation_losses = checkpoint.get("validation_losses", None)
        if checkpoint.get("validation_epochs", None) != None:
            validation_epochs = checkpoint.get("validation_epochs", None)
        if checkpoint.get("validation_accuracy", None) != None:
            val_acc = checkpoint.get("validation_accuracy", None)
        if checkpoint.get("train_accuracy", None) != None:
            train_acc = checkpoint.get("train_accuracy", None)

    for i, r in enumerate(learning_rates):
        writer.add_scalar("Metric/lr", r, i)
    for i, r in enumerate(train_acc):
        writer.add_scalar("Metric/train_acc", r, i)
    for i, r in enumerate(val_acc):
        writer.add_scalar("Metric/val_acc", r, validation_epochs[i])
    for i, l in enumerate(train_losses):
        writer.add_scalar("Loss/train", l, i)
    for i, l in enumerate(validation_losses):
        writer.add_scalar("Loss/val", l, validation_epochs[i])

    model.train()
    # best_model_yet = model_path
    # best_loss_yet = None
    loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for e in range(epochs):
        y_true = torch.empty(0, device=device)
        y_out = torch.empty((0, num_outputs), device=device)
        cum_loss = 0
        cnt = 0
        for inputs, targets in tqdm(iter(loader)):
            inputs = inputs.to(device)
            if isinstance(loss, BCEWithLogitsLoss):
                targets = one_hot(targets, num_outputs).float()
            targets = targets.to(device)

            y_true = torch.cat((y_true, targets), 0)
            optimizer.zero_grad()
            logits = model(inputs)
            l = loss(logits, targets)
            y_out = torch.cat((y_out, logits.detach().clone()), 0)
            if math.isnan(l):
                if not os.path.isdir("./broken_model"):
                    os.mkdir("broken_model")
                save_dict = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": base_epoch + e,
                    "learning_rates": learning_rates,
                    "train_losses": train_losses,
                    "validation_losses": validation_losses,
                    "validation_epochs": validation_epochs,
                    "validation_accuracy": val_acc,
                    "ds_type": dataset_type,
                    "train_accuracy": train_acc,
                }
                path = os.path.join(output_dir, f"broken_model_{base_epoch + e}")
                torch.save(save_dict, path)
                print("NaN loss")
                exit()
            l.backward()
            optimizer.step()
            cum_loss = cum_loss + l
            cnt = cnt + inputs.shape[0]
        # y_out = torch.softmax(y_out, dim=1)
        y_pred = torch.argmax(y_out, dim=1)
        # y_true = y_true.cpu().numpy()
        # y_out = y_out.cpu().numpy()
        # y_pred = y_pred.cpu().numpy()
        train_loss = cum_loss.detach().cpu()
        acc = (y_true == y_pred).sum() / y_out.shape[0]
        train_acc.append(acc)
        print(f"train accuracy: {acc}")
        writer.add_scalar("Metric/train_acc", acc, base_epoch + e)
        train_losses.append(train_loss)
        writer.add_scalar("Loss/train", train_loss, base_epoch + e)
        print(f"Epoch {e + 1}/{epochs} loss: {cum_loss}")  # / cnt}")
        print("\n==============\n")
        learning_rates.append(scheduler.get_lr())
        scheduler.step()
        if (e + 1) % save_each == 0:
            validation_epochs.append(e)
            save_dict = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epoch": base_epoch + e,
                "train_losses": train_losses,
                "validation_losses": validation_losses,
                "validation_epochs": validation_epochs,
                "validation_accuracy": val_acc,
                "learning_rates": learning_rates,
                "train_accuracy": train_acc,
            }
            if dataset_type == "shortcut":
                save_dict["shortcut_indices"] = train_set.transform_indices
            elif dataset_type == "mislabeled":
                save_dict["mislabeled_indices"] = train_set.transform_indices

            save_id = f"{save_id_base}_{base_epoch + e}"
            model_save_path = os.path.join(output_dir, save_id)
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            torch.save(save_dict, model_save_path)
            saved_files.append((model_save_path, save_id))

            valeval = evaluate_model(
                model=model, device=device, num_outputs=num_outputs, batch_size=batch_size, val_set=val_set
            )
            print(f"validation accuracy: {valeval}")
            writer.add_scalar("Metric/val_acc", valeval, base_epoch + e)
            val_acc.append(valeval)
            # if best_loss_yet is None or best_loss_yet > validation_loss:
            #    best_loss_yet = validation_loss
            #    path = os.path.join(output_dir, f"best_val_score_{dataset_name}_{model_name}_{base_epoch + e}")
            #    torch.save(save_dict, path)
            #    if best_model_yet is not None:
            #        os.remove(best_model_yet)
            #    best_model_yet = path
        writer.flush()

        # Save train and validation loss figures
        # plt.subplot(2, 1, 1)
        # plt.title("Training Loss")
        # plt.plot(base_epoch + np.asarray(range(epochs)), np.asarray(train_losses))
        # plt.subplot(2, 1, 2)
        # plt.title("Validation Loss")
        # plt.plot(base_epoch + np.asarray(vaidation_epochs), np.asarray(validation_losses))
        # plt.savefig(os.path.join(output_dir, f"{dataset_name}_{model_name}_{base_epoch + epochs}_losses.png"))
    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Define argument for method with choices
    parser.add_argument("--dataset_type", required=True, choices=["vanilla", "mislabeled", "shortcut", "mixed", "subclass"])

    # Define other required arguments
    parser.add_argument("--tiny_imgnet_path", required=True, type=str, help="Path to Tiny ImageNet dataset")
    parser.add_argument("--model_path", required=False, type=str, default=None, help="Path to Tiny ImageNet dataset")
    parser.add_argument("--metadata_path", required=False, default=None)
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save outputs")
    parser.add_argument("--num_groups", required=False, type=int, default=None)
    parser.add_argument("--epochs", required=True, type=int, default=100)
    parser.add_argument("--save_each", required=True, type=int, default=10)
    parser.add_argument("--batch_size", required=True, type=int, default=64)
    parser.add_argument("--base_epoch", required=False, type=int, default=0)
    parser.add_argument("--lr", required=True, type=float, default=0.1)
    parser.add_argument("--download", action="store_true", help="Download the datasets and checkpoints")
    parser.add_argument("--device", required=True, type=str, help="Device to run the model on", choices=["cpu", "cuda"])
    parser.add_argument("--pretrained", action="store_true")

    parser.add_argument("--optimizer", required=False, type=str, default="adam")
    parser.add_argument("--weight_decay", required=False, type=float, default=0.0)
    parser.add_argument("--momentum", required=False, type=float, default=0.0)
    parser.add_argument("--scheduler", required=False, type=str, default="constant")
    parser.add_argument("--loss", required=False, type=str, default="cross_entropy")
    parser.add_argument("--augmentation", required=False, type=str, default=None)

    args = parser.parse_args()

    assert not (args.num_groups is None and args.dataset_type == "subclass")
    assert not (args.num_groups is not None and args.dataset_type != "subclass")

    # Call the function with parsed arguments
    train_model(
        args.dataset_type,
        args.tiny_imgnet_path,
        args.metadata_path,
        args.output_dir,
        args.download,
        args.num_groups,
        args.device,
        args.pretrained,
        args.epochs,
        args.lr,
        args.batch_size,
        args.save_each,
        args.optimizer,
        args.weight_decay,
        args.momentum,
        args.scheduler,
        args.loss,
        args.augmentation,
        args.model_path,
        args.base_epoch,
    )
