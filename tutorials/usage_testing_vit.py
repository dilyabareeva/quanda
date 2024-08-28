"Larhe chunks of code borrowed from https://github.com/unlearning-challenge/starting-kit/blob/main/unlearning-CIFAR10.ipynb"
import copy
from multiprocessing import freeze_support

import lightning as L
import matplotlib.pyplot as plt
import torch
import torchvision

# from torch import nn
# from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.utils import make_grid
from tqdm import tqdm

from quanda.explainers.wrappers import CaptumArnoldi, captum_similarity_explain
from quanda.metrics.localization import ClassDetectionMetric
from quanda.metrics.randomization import ModelRandomizationMetric
from quanda.metrics.unnamed import DatasetCleaningMetric, TopKOverlapMetric
from quanda.toy_benchmarks.localization import SubclassDetection
from quanda.utils.training import BasicLightningModule
from tutorials.utils.datasets import AnnotatedDataset, TrainTinyImageNetDataset

DEVICE = "cuda:0"  # "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")

print("Running on device:", DEVICE.upper())

# manual random seed is used for dataset partitioning
# to ensure reproducible results across runs
RNG = torch.Generator().manual_seed(42)


# minimal custom torch dataset that places the data on the specified device
class OnDeviceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, device: str):
        self.dataset = dataset
        self.device = device

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data.to(self.device), torch.tensor(target).to(self.device)

    def __len__(self):
        return len(self.dataset)


def hf_output_ce_loss(outputs, targets):
    return torch.nn.CrossEntropyLoss(reduction="none")(outputs.logits, targets)


def main():
    # ++++++++++++++++++++++++++++++++++++++++++
    # #Download dataset and pre-trained model
    # ++++++++++++++++++++++++++++++++++++++++++
    torch.set_float32_matmul_precision("medium")

    n_classes = 200
    batch_size = 64
    num_workers = 8
    data_path = "/home/bareeva/Projects/data_attribution_evaluation/assets/tiny-imagenet-200"
    rng = torch.Generator().manual_seed(42)

    transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    train_set = TrainTinyImageNetDataset(local_path=data_path, transforms=transform)
    train_set = OnDeviceDataset(train_set, DEVICE)
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    hold_out = AnnotatedDataset(local_path=data_path, transforms=transform)
    test_set, val_set = torch.utils.data.random_split(hold_out, [0.5, 0.5], generator=rng)
    test_set, val_set = OnDeviceDataset(test_set, DEVICE), OnDeviceDataset(val_set, DEVICE)

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = resnet18(pretrained=False, num_classes=n_classes)

    # download pre-trained weights
    local_path = "/home/bareeva/Projects/data_attribution_evaluation/assets/tiny_imagenet_resnet18.pth"

    weights_pretrained = torch.load(local_path, map_location=DEVICE)
    model.load_state_dict(weights_pretrained)

    init_model = resnet18(pretrained=False, num_classes=n_classes)

    model.to(DEVICE)
    init_model.to(DEVICE)
    model.eval()
    init_model.eval()

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
            _, predicted = torch.max(outputs.logits, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return correct / total

    # print(f"Train set accuracy: {100.0 * accuracy(model, train_dataloader):0.1f}%")
    # print(f"Test set accuracy: {100.0 * accuracy(model, test_dataloader):0.1f}%")

    # ++++++++++++++++++++++++++++++++++++++++++
    # Computing metrics while generating explanations
    # ++++++++++++++++++++++++++++++++++++++++++

    explain = captum_similarity_explain
    explainer_cls = CaptumArnoldi
    explain_fn_kwargs = {
        "projection_on_cpu": False,
        "loss_fn": torch.nn.CrossEntropyLoss(reduction="none"),
        "arnoldi_tol": 1e-2,
        "batch_size": 32,
        "projection_dim": 10,
        "arnoldi_dim": 10,
        "checkpoint": local_path,
    }
    model_id = "default_model_id"
    cache_dir = "./cache"
    model_rand = ModelRandomizationMetric(
        model=model,
        train_dataset=train_set,
        explainer_cls=explainer_cls,
        expl_kwargs=explain_fn_kwargs,
        model_id=model_id,
        cache_dir=cache_dir,
        correlation_fn="spearman",
        seed=42,
    )

    id_class = ClassDetectionMetric(model=model, train_dataset=train_set, device=DEVICE)

    top_k = TopKOverlapMetric(model=model, train_dataset=train_set, top_k=1, device=DEVICE)

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
        device=DEVICE,
    )

    # iterate over test set and feed tensor batches first to explain, then to metric
    for i, (data, target) in enumerate(tqdm(test_dataloader)):
        data, target = data.to(DEVICE), target.to(DEVICE)
        tda = explain(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            test_tensor=data,
            train_dataset=train_set,
            device=DEVICE,
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

    print(f"Test set accuracy: {100.0 * accuracy(model, test_dataloader):0.1f}%")

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
        device=DEVICE,
    )

    score = bench.evaluate(
        expl_dataset=test_set,
        explainer_cls=explainer_cls,
        expl_kwargs=explain_fn_kwargs,
        cache_dir="./cache",
        model_id="default_model_id",
    )

    print("Subclass Detection Benchmark Score:", score)


if __name__ == "__main__":
    freeze_support()
    main()
