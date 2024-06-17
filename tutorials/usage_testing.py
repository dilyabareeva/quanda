"Larhe chunks of code borrowed from https://github.com/unlearning-challenge/starting-kit/blob/main/unlearning-CIFAR10.ipynb"

import os
from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import requests
import torch
import torchvision

# from torch import nn
# from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.utils import make_grid
from tqdm import tqdm

from src.metrics.randomization.model_randomization import (
    ModelRandomizationMetric,
)
from src.utils.explain_wrapper import explain
from src.utils.functions.similarities import cosine_similarity

DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", DEVICE.upper())

# manual random seed is used for dataset partitioning
# to ensure reproducible results across runs
RNG = torch.Generator().manual_seed(42)

# ++++++++++++++++++++++++++++++++++++++++++
# #Download dataset and pre-trained model
# ++++++++++++++++++++++++++++++++++++++++++


def main():
    # download and pre-process CIFAR10
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(root="./tutorials/data", train=True, download=True, transform=normalize)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(root="./tutorials/data", train=False, download=True, transform=normalize)
    test_set, val_set = torch.utils.data.random_split(held_out, [0.5, 0.5], generator=RNG)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    # val_loader = DataLoader(val_set, batch_size=128, shuffle=False, num_workers=2)

    # download pre-trained weights
    local_path = "./tutorials/model_weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get("https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth")
        open(local_path, "wb").write(response.content)

    weights_pretrained = torch.load(local_path, map_location=DEVICE)

    # load model with pre-trained weights
    model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)
    model.eval()

    # a temporary data loader without normalization, just to show the images
    tmp_dl = DataLoader(
        torchvision.datasets.CIFAR10(root="./tutorials/data", train=True, download=True, transform=transforms.ToTensor()),
        batch_size=16 * 5,
        shuffle=False,
    )
    images, labels = next(iter(tmp_dl))

    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title("Sample images from CIFAR10 dataset")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    plt.show()

    def accuracy(net, loader):
        """Return accuracy on a dataset given by the data loader."""
        correct = 0
        total = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct / total

    print(f"Train set accuracy: {100.0 * accuracy(model, train_loader):0.1f}%")
    print(f"Test set accuracy: {100.0 * accuracy(model, test_loader):0.1f}%")

    # ++++++++++++++++++++++++++++++++++++++++++
    # Training configuration
    # ++++++++++++++++++++++++++++++++++++++++++
    """
    max_epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD
    optimizer_kwargs = {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs = {"T_max": max_epochs}
    """
    # ++++++++++++++++++++++++++++++++++++++++++
    # Computing metrics while generating explanations
    # ++++++++++++++++++++++++++++++++++++++++++

    metric = ModelRandomizationMetric(
        model=model,
        train_dataset=train_set,
        explain_fn=explain,
        explain_fn_kwargs={"method": "SimilarityInfluence", "layer": "avgpool"},
        model_id="default_model_id",
        cache_dir="./cache",
        correlation_fn="spearman",
        seed=42,
        device="cpu",
    )

    # iterate over test set and feed tensor batches first to explain, then to metric
    for i, (data, target) in enumerate(tqdm(test_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        tda = explain(
            model=model,
            model_id="default_model_id",
            cache_dir="./cache",
            method="SimilarityInfluence",
            train_dataset=train_set,
            test_tensor=data,
            layer="avgpool",
            similarity_metric=cosine_similarity,
            similarity_direction="max",
            batch_size=1,
        )
        metric.update(data, tda)

    print("Model randomization metric output:", metric.compute().item())
    print(f"Test set accuracy: {100.0 * accuracy(model, test_loader):0.1f}%")


if __name__ == "__main__":
    freeze_support()
    main()
