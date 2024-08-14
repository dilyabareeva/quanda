"Larhe chunks of code borrowed from https://github.com/unlearning-challenge/starting-kit/blob/main/unlearning-CIFAR10.ipynb"
import copy
import os
from multiprocessing import freeze_support

import lightning as L
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

from quanda.explainers.wrappers import (
    CaptumSimilarity,
    captum_similarity_explain,
)
from quanda.metrics.localization import ClassDetectionMetric
from quanda.metrics.randomization import ModelRandomizationMetric
from quanda.metrics.unnamed import DatasetCleaningMetric, TopKOverlapMetric
from quanda.toy_benchmarks.localization import SubclassDetection
from quanda.utils.training import BasicLightningModule

DEVICE = "cuda:0"  # "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")

print("Running on device:", DEVICE.upper())

# manual random seed is used for dataset partitioning
# to ensure reproducible results across runs
RNG = torch.Generator().manual_seed(42)


def main():
    # ++++++++++++++++++++++++++++++++++++++++++
    # #Download dataset and pre-trained model
    # ++++++++++++++++++++++++++++++++++++++++++

    # download and pre-process CIFAR10
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(root="./tutorials/data", train=True, download=True, transform=normalize)
    train_dataloader = DataLoader(train_set, batch_size=100, shuffle=True, num_workers=8)

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(root="./tutorials/data", train=False, download=True, transform=normalize)
    test_set, val_set = torch.utils.data.random_split(held_out, [0.1, 0.9], generator=RNG)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False, num_workers=8)
    # val_dataloader = DataLoader(val_set, batch_size=100, shuffle=False, num_workers=8)

    # download pre-trained weights
    local_path = "./tutorials/model_weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get("https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth")
        open(local_path, "wb").write(response.content)

    weights_pretrained = torch.load(local_path, map_location=DEVICE)

    # load model with pre-trained weights
    model = resnet18(weights=None, num_classes=10)
    init_model = resnet18(weights=None, num_classes=10)
    model.load_state_dict(weights_pretrained)
    model.to(DEVICE)

    device = "cpu"
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

    print(f"Train set accuracy: {100.0 * accuracy(model, train_dataloader):0.1f}%")
    print(f"Test set accuracy: {100.0 * accuracy(model, test_loader):0.1f}%")

    # ++++++++++++++++++++++++++++++++++++++++++
    # Computing metrics while generating explanations
    # ++++++++++++++++++++++++++++++++++++++++++

    explain = captum_similarity_explain
    explain_fn_kwargs = {"layers": "avgpool", "batch_size": 100}
    model_id = "default_model_id"
    cache_dir = "./cache"
    model_rand = ModelRandomizationMetric(
        model=model,
        train_dataset=train_set,
        explainer_cls=CaptumSimilarity,
        expl_kwargs=explain_fn_kwargs,
        model_id=model_id,
        cache_dir=cache_dir,
        correlation_fn="spearman",
        seed=42,
        device=device,
    )

    id_class = ClassDetectionMetric(model=model, train_dataset=train_set, device=device)

    top_k = TopKOverlapMetric(model=model, train_dataset=train_set, top_k=1, device=device)

    # dataset cleaning
    max_epochs = 1
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD
    lr = 0.1
    optimizer_kwargs = {"momentum": 0.9, "weight_decay": 5e-4}
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR
    scheduler_kwargs = {"T_max": max_epochs}

    pl_module = BasicLightningModule(
        model=model,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        lr=lr,
        criterion=criterion,
    )

    init_pl_module = BasicLightningModule(
        model=init_model,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
        lr=lr,
        criterion=criterion,
    )

    trainer = L.Trainer(max_epochs=max_epochs)

    data_clean = DatasetCleaningMetric(
        model=pl_module,
        init_model=copy.deepcopy(init_pl_module),
        train_dataset=train_set,
        global_method="sum_abs",
        trainer=trainer,
        top_k=50,
        device=device,
    )

    # iterate over test set and feed tensor batches first to explain, then to metric
    for i, (data, target) in enumerate(tqdm(test_loader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        tda = explain(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            test_tensor=data,
            train_dataset=train_set,
            device=device,
            **explain_fn_kwargs,
        )
        model_rand.update(data, tda)
        id_class.update(target, tda)
        top_k.update(tda)
        data_clean.update(tda)

    print("Model randomization metric output:", model_rand.compute())
    print("Identical class metric output:", id_class.compute())
    print("Top-k overlap metric output:", top_k.compute())

    print("Dataset cleaning metric computation started...")
    print("Dataset cleaning metric output:", data_clean.compute())

    print(f"Test set accuracy: {100.0 * accuracy(model, test_loader):0.1f}%")

    # ++++++++++++++++++++++++++++++++++++++++++
    # Subclass Detection Benchmark Generation and Evaluation
    # ++++++++++++++++++++++++++++++++++++++++++

    trainer = L.Trainer(max_epochs=max_epochs)

    bench = SubclassDetection.generate(
        model=copy.deepcopy(init_pl_module),
        train_dataset=train_set,
        trainer=trainer,
        val_dataset=val_set,
        n_classes=10,
        n_groups=2,
        class_to_group="random",
        seed=42,
        batch_size=100,
        device=device,
    )

    score = bench.evaluate(
        expl_dataset=test_set,
        explainer_cls=CaptumSimilarity,
        expl_kwargs={"layers": "model.avgpool", "batch_size": 100},
        cache_dir="./cache",
        model_id="default_model_id",
    )

    print("Subclass Detection Benchmark Score:", score)


if __name__ == "__main__":
    freeze_support()
    main()
