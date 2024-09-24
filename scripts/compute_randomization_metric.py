import logging
import os
import random
import subprocess
from argparse import ArgumentParser

import lightning as L
import torch
import torchvision.transforms as transforms
import wandb
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Subset

from quanda.explainers import RandomExplainer
from quanda.explainers.wrappers import (
    TRAK,
    CaptumArnoldi,
    CaptumSimilarity,
    CaptumTracInCPFast,
    RepresenterPoints,
)
from quanda.metrics.heuristics import ModelRandomizationMetric
from quanda.utils.cache import ExplanationsCache as EC
from quanda.utils.datasets.transformed import (
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


load_dotenv()


def compute_randomization_metric(
    method, tiny_in_path, panda_sketch_path, explanations_dir, checkpoints_dir, metadata_dir, download
):
    torch.set_float32_matmul_precision("medium")

    # Initialize WandbLogger
    wandb.init(project="quanda", name="tiny_inet_resnet18_big_eval")

    # Downloading the datasets and checkpoints

    # We first download the datasets (uncomment the following cell if you haven't downloaded the datasets yet).:
    os.makedirs(explanations_dir, exist_ok=True)

    if download:
        os.makedirs(metadata_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        # os.makedirs(tiny_in_path, exist_ok=True)

        # subprocess.run(["wget", "-qP", tiny_in_path, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])
        # subprocess.run(["unzip", "-qq", os.path.join(tiny_in_path, "tiny-imagenet-200.zip"), "-d", tiny_in_path])
        subprocess.run(
            ["wget", "-qP", metadata_dir, "https://datacloud.hhi.fraunhofer.de/s/FpPWkzPmM3s9ZqF/download/sketch.zip"]
        )
        subprocess.run(["unzip", "-qq", os.path.join(metadata_dir, "sketch.zip"), "-d", metadata_dir])

        # Next we download all the necessary checkpoints and the dataset metadata
        subprocess.run(
            [
                "wget",
                "-P",
                checkpoints_dir,
                "https://datacloud.hhi.fraunhofer.de/s/ZE5dBnfzW94Xkoo/download/tiny_inet_resnet18.zip",
            ]
        )
        subprocess.run(["unzip", "-qq", "-j", os.path.join(checkpoints_dir, "tiny_inet_resnet18.zip"), "-d", metadata_dir])
        subprocess.run(
            ["wget", "-qP", metadata_dir, "https://datacloud.hhi.fraunhofer.de/s/AmnCXAC8zx3YQgP/download/dataset_indices.zip"]
        )
        subprocess.run(["unzip", "-qq", "-j", os.path.join(metadata_dir, "dataset_indices.zip"), "-d", metadata_dir])

    n_epochs = 10
    checkpoints = [
        os.path.join(checkpoints_dir, f"tiny_imagenet_resnet18_epoch={epoch:02d}.ckpt") for epoch in range(1, n_epochs, 2)
    ]

    # Dataset Construction

    # Loading the dataset metadata
    class_to_group = torch.load(os.path.join(metadata_dir, "class_to_group.pth"))
    test_split = torch.load(os.path.join(metadata_dir, "test_indices.pth"))
    panda_train_indices = torch.load(os.path.join(metadata_dir, "panda_train_indices.pth"))

    n_classes = 200
    new_n_classes = len(set(list(class_to_group.values())))
    batch_size = 64
    num_workers = 1
    device = "cuda:0"

    generator = random.Random(27)

    # Define transformations
    regular_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    # Load the TinyImageNet dataset
    tiny_in_path = os.path.join(tiny_in_path, "tiny-imagenet-200/")
    with open(tiny_in_path + "wnids.txt", "r") as f:
        id_dict = {line.strip(): i for i, line in enumerate(f)}

    val_annotations = {}
    with open(tiny_in_path + "val/val_annotations.txt", "r") as f:
        val_annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}

    train_set = CustomDataset(tiny_in_path + "train", classes=list(id_dict.keys()), classes_to_idx=id_dict, transform=None)
    holdout_set = AnnotatedDataset(
        local_path=tiny_in_path + "val", transforms=None, id_dict=id_dict, annotation=val_annotations
    )
    test_set = torch.utils.data.Subset(holdout_set, test_split)

    backdoor_transforms = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    panda_dataset = CustomDataset(
        panda_sketch_path, classes=["n02510455"], classes_to_idx={"n02510455": 5}, transform=backdoor_transforms
    )
    panda_set = torch.utils.data.Subset(panda_dataset, panda_train_indices)

    def add_yellow_square(img):
        square_size = (15, 15)  # Size of the square
        yellow_square = Image.new("RGB", square_size, (255, 255, 0))  # Create a yellow square
        img.paste(yellow_square, (10, 10))  # Paste it onto the image at the specified position
        return img

    train_set = special_dataset(
        train_set,
        n_classes,
        new_n_classes,
        regular_transforms,
        class_to_group=class_to_group,
        shortcut_fn=add_yellow_square,
        backdoor_dataset=panda_set,
        shortcut_transform_indices=torch.load(os.path.join(metadata_dir, "all_train_shortcut_indices_for_generation.pth")),
        flipping_transform_dict=torch.load(os.path.join(metadata_dir, "all_train_flipped_dict_for_generation.pth")),
    )

    test_set_grouped = LabelGroupingDataset(
        dataset=test_set,
        n_classes=n_classes,
        dataset_transform=regular_transforms,
        class_to_group=class_to_group,
    )

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    lit_model = LitModel.load_from_checkpoint(
        checkpoints[-1],
        n_batches=len(train_dataloader),
        num_labels=new_n_classes,
        device=device,
        map_location=torch.device(device),
    )
    lit_model.eval()

    def load_state_dict(module: L.LightningModule, path: str) -> int:
        checkpoints = torch.load(path, map_location=torch.device("cuda:0"))
        module.model.load_state_dict(checkpoints["model_state_dict"])
        module.eval()
        return module.lr

    # Dataloader for Model Randomization, Top-K Overlap
    clean_samples = torch.load(os.path.join(metadata_dir, "big_eval_test_clean_indices.pth"))
    clean_dataset = torch.utils.data.Subset(test_set_grouped, clean_samples)
    dataloader = torch.utils.data.DataLoader(clean_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model_id = "0"
    randomization_dir = os.path.join(explanations_dir, "randomization_explanations")

    if method == "similarity":
        cache_dir = os.path.join(randomization_dir, method)
        os.makedirs(cache_dir, exist_ok=True)

        # Initialize Explainer
        explainer_cls = CaptumSimilarity
        explain_kwargs = {
            "layers": "model.avgpool",
            "similarity_metric": cosine_similarity,
            "batch_size": batch_size,
            "load_from_disk": False,
            "model_id": model_id,
            "cache_dir": cache_dir,
        }

    if method == "representer_points":
        cache_dir = os.path.join(randomization_dir, method)
        os.makedirs(cache_dir, exist_ok=True)

        explainer_cls = RepresenterPoints
        explain_kwargs = {
            "features_layer": "model.avgpool",
            "classifier_layer": "model.fc",
            "batch_size": batch_size,
            "features_postprocess": lambda x: x[:, :, 0, 0],
            "load_from_disk": False,
            "show_progress": False,
            "model_id": model_id,
            "cache_dir": cache_dir,
        }

    if method == "tracincpfast":

        explainer_cls = CaptumTracInCPFast
        explain_kwargs = {
            "checkpoints": checkpoints,
            "checkpoints_load_func": load_state_dict,
            "loss_fn": torch.nn.CrossEntropyLoss(reduction="mean"),
            "final_fc_layer": "model.fc",
            "device": device,
            "batch_size": batch_size * 4,
        }

    if method == "arnoldi":

        train_dataset = train_dataloader.dataset
        num_samples = 5000
        indices = generator.sample(range(len(train_dataset)), num_samples)
        hessian_dataset = Subset(train_dataset, indices)

        explainer_cls = CaptumArnoldi
        explain_kwargs = {
            "hessian_dataset": hessian_dataset,
            "checkpoint": checkpoints[-1],
            "loss_fn": torch.nn.CrossEntropyLoss(reduction="none"),
            "checkpoints_load_func": load_state_dict,
            "projection_dim": 100,
            "arnoldi_dim": 200,
            "batch_size": batch_size * 4,
            "layers": ["model.fc"],  # only the last layer
            "device": device,
        }

    if method == "trak":
        cache_dir = os.path.join(randomization_dir, method)
        os.makedirs(cache_dir, exist_ok=True)

        explainer_cls = TRAK
        explain_kwargs = {
            "model_id": model_id,
            "cache_dir": cache_dir,
            "projector": "cuda",
            "proj_dim": 4096,
            "load_from_disk": False,
        }

    if method == "random":
        explainer_cls = RandomExplainer
        explain_kwargs = {"seed": 28}

    model_rand = ModelRandomizationMetric(
        model=lit_model,
        train_dataset=train_set,
        explainer_cls=explainer_cls,
        expl_kwargs=explain_kwargs,
        correlation_fn="spearman",
        seed=42,
    )

    method_save_dir = os.path.join(explanations_dir, method)
    subset_save_dir = os.path.join(method_save_dir, "randomization")
    explanations = EC.load(subset_save_dir)

    for i, (test_tensor, test_labels) in enumerate(dataloader):
        test_tensor, test_labels = test_tensor.to(device), test_labels.to(device)
        explanation_targets = torch.tensor(
            [lit_model.model(test_tensor.to(device)).argmax().item() for i in range(len(test_tensor))]
        )
        model_rand.update(test_tensor, explanations[i], explanation_targets)

    score = model_rand.compute()
    wandb.log({f"{method}_randomization": score})


if __name__ == "__main__":
    parser = ArgumentParser()

    # Define argument for method with choices
    parser.add_argument(
        "--method",
        required=True,
        choices=["similarity", "representer_points", "tracincpfast", "arnoldi", "trak", "random"],
        help="Choose the explanation method to use.",
    )

    # Define other required arguments
    parser.add_argument("--tiny_in_path", required=True, type=str, help="Path to Tiny ImageNet dataset")
    parser.add_argument("--panda_sketch_path", required=True, type=str, help="Path to ImageNet-Sketch dataset")
    parser.add_argument("--explanations_dir", required=True, type=str, help="Directory to save outputs")
    parser.add_argument("--checkpoints_dir", required=True, type=str, help="Directory to checkpoints")
    parser.add_argument("--metadata_dir", required=True, type=str, help="Directory to metadata")
    parser.add_argument("--download", action="store_true", help="Download the datasets and checkpoints")
    args = parser.parse_args()

    # Call the function with parsed arguments
    compute_randomization_metric(
        args.method,
        args.tiny_in_path,
        args.panda_sketch_path,
        args.explanations_dir,
        args.checkpoints_dir,
        args.metadata_dir,
        args.download,
    )
