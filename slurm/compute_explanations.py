import logging
import os
import random
import subprocess
from argparse import ArgumentParser

import lightning as L
import torch
import torchvision.transforms as transforms
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Subset

from quanda.explainers.wrappers import (
    TRAK,
    CaptumArnoldi,
    CaptumSimilarity,
    CaptumTracInCPFast,
    RepresenterPoints,
)
from quanda.utils.cache import ExplanationsCache as EC
from quanda.utils.functions import cosine_similarity
from tutorials.utils.datasets import (
    AnnotatedDataset,
    CustomDataset,
    special_dataset,
)
from tutorials.utils.modules import LitModel

logger = logging.getLogger(__name__)


def compute_explanations(method, tiny_in_path, panda_sketch_path, output_dir, checkpoints_dir, metadata_dir, download):
    torch.set_float32_matmul_precision("medium")

    # Downloading the datasets and checkpoints

    # We first download the datasets (uncomment the following cell if you haven't downloaded the datasets yet).:
    os.makedirs(output_dir, exist_ok=True)

    if download:
        os.makedirs(metadata_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(tiny_in_path, exist_ok=True)

        subprocess.run(["wget", "-P", tiny_in_path, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])
        subprocess.run(["unzip", os.path.join(tiny_in_path, "tiny-imagenet-200.zip"), "-d", tiny_in_path])
        subprocess.run(
            ["wget", "-P", metadata_dir, "https://datacloud.hhi.fraunhofer.de/s/FpPWkzPmM3s9ZqF/download/sketch.zip"]
        )
        subprocess.run(["unzip", os.path.join(metadata_dir, "sketch.zip"), "-d", metadata_dir])

        # Next we download all the necessary checkpoints and the dataset metadata
        subprocess.run(
            [
                "wget",
                "-P",
                checkpoints_dir,
                "https://datacloud.hhi.fraunhofer.de/s/ZE5dBnfzW94Xkoo/download/tiny_inet_resnet18.zip",
            ]
        )
        subprocess.run(["unzip", "-j", os.path.join(checkpoints_dir, "tiny_inet_resnet18.zip"), "-d", metadata_dir])
        subprocess.run(
            ["wget", "-P", metadata_dir, "https://datacloud.hhi.fraunhofer.de/s/AmnCXAC8zx3YQgP/download/dataset_indices.zip"]
        )
        subprocess.run(["unzip", "-j", os.path.join(metadata_dir, "dataset_indices.zip"), "-d", metadata_dir])

    n_epochs = 10
    checkpoints = [
        os.path.join(checkpoints_dir, f"tiny_imagenet_resnet18_epoch={epoch:02d}.ckpt") for epoch in range(1, n_epochs, 2)
    ]

    # Dataset Construction

    # Loading the dataset metadata
    class_to_group = torch.load(os.path.join(metadata_dir, "class_to_group.pth"))
    test_split = torch.load(os.path.join(metadata_dir, "test_indices.pth"))

    # Optional: load environmental variable from .env file (incl. wandb api key)
    load_dotenv()

    n_classes = 200
    new_n_classes = len(set(list(class_to_group.values())))
    batch_size = 64
    num_workers = 8

    torch_rng = torch.Generator().manual_seed(27)
    generator = random.Random(27)

    # Define transformations
    regular_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    # Load the TinyImageNet dataset
    id_dict = {}
    with open(tiny_in_path + "/wnids.txt", "r") as f:
        id_dict = {line.strip(): i for i, line in enumerate(f)}

    val_annotations = {}
    with open(tiny_in_path + "/val/val_annotations.txt", "r") as f:
        val_annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}

    train_set = CustomDataset(tiny_in_path + "/train", classes=list(id_dict.keys()), classes_to_idx=id_dict, transform=None)
    holdout_set = AnnotatedDataset(
        local_path=tiny_in_path + "/val", transforms=None, id_dict=id_dict, annotation=val_annotations
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
    panda_set, panda_dataset = torch.utils.data.random_split(panda_dataset, [30, len(panda_dataset) - 30], generator=torch_rng)
    panda_val, panda_dataset = torch.utils.data.random_split(panda_dataset, [10, len(panda_dataset) - 10], generator=torch_rng)
    panda_test, _ = torch.utils.data.random_split(panda_dataset, [10, len(panda_dataset) - 10], generator=torch_rng)

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

    test_set = special_dataset(
        test_set,
        n_classes,
        new_n_classes,
        regular_transforms,
        class_to_group=class_to_group,
        shortcut_fn=add_yellow_square,
        backdoor_dataset=panda_test,
        shortcut_transform_indices=torch.load(os.path.join(metadata_dir, "all_test_shortcut_indices_for_generation.pth")),
        flipping_transform_dict={},
    )

    test_indices = (
        torch.load(os.path.join(metadata_dir, "big_eval_test_backdoor_indices.pth"))
        + torch.load(os.path.join(metadata_dir, "big_eval_test_shortcut_indices.pth"))
        + torch.load(os.path.join(metadata_dir, "big_eval_test_dogs_indices.pth"))
        + torch.load(os.path.join(metadata_dir, "big_eval_test_cats_indices.pth"))
        + torch.load(os.path.join(metadata_dir, "big_eval_test_clean_indices.pth"))
    )
    target_test_set = torch.utils.data.Subset(test_set, test_indices)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(
        target_test_set, batch_size=len(test_indices), shuffle=False, num_workers=num_workers
    )

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    images, labels = next(iter(test_dataloader))
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.title("Sample images from CIFAR10 dataset")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    plt.show()

    lit_model = LitModel.load_from_checkpoint(
        checkpoints[-1], n_batches=len(train_dataloader), num_labels=new_n_classes, map_location=torch.device("cuda:0")
    )
    lit_model.model = lit_model.model.eval()

    if method == "similarity":
        # Initialize Explainer
        explainer_similarity = CaptumSimilarity(
            model=lit_model,
            model_id="0",
            cache_dir="tmp",
            train_dataset=train_dataloader.dataset,
            layers="model.avgpool",
            similarity_metric=cosine_similarity,
            device="cuda:0",
            batch_size=10,
            load_from_disk=True,
        )

        method_save_dir = os.path.join(output_dir, "expl_similarity")
        os.makedirs(method_save_dir, exist_ok=True)
        # Explain test samples
        logger.info("Explaining test samples")
        for i, (test_tensor, test_labels) in enumerate(test_dataloader):
            explanation_targets = [
                lit_model.model(test_tensor[i].unsqueeze(0).to("cuda:0")).argmax().item() for i in range(len(test_tensor))
            ]
            explanations_similarity = explainer_similarity.explain(test_tensor)
            EC.save(method_save_dir, explanations_similarity, i)

    if method == "representer_points":
        explainer_repr = RepresenterPoints(
            model=lit_model,
            cache_dir="tmp_repr",
            train_dataset=train_dataloader.dataset,
            features_layer="model.avgpool",
            classifier_layer="model.fc",
            batch_size=32,
            features_postprocess=lambda x: x[:, :, 0, 0],
            model_id="demo",
            load_from_disk=False,
            show_progress=False,
        )

        method_save_dir = os.path.join(output_dir, "expl_representer_points")
        os.makedirs(method_save_dir, exist_ok=True)
        # Explain test samples
        logger.info("Explaining test samples")
        for i, (test_tensor, test_labels) in enumerate(test_dataloader):
            explanation_targets = [
                lit_model.model(test_tensor[i].unsqueeze(0).to("cuda:0")).argmax().item() for i in range(len(test_tensor))
            ]
            explanations_repr = explainer_repr.explain(test_tensor, explanation_targets)
            EC.save(method_save_dir, explanations_repr, i)

    if method == "tracincpfast":

        def load_state_dict(module: L.LightningModule, path: str) -> int:
            module = type(module).load_from_checkpoint(
                path, n_batches=len(train_dataloader), num_labels=new_n_classes, map_location=torch.device("cuda:0")
            )
            module.model.eval()
            return module.lr

        # Initialize Explainer
        logger.info("Explaining test samples")
        explainer_tracincpfast = CaptumTracInCPFast(
            model=lit_model,
            train_dataset=train_dataloader.dataset,
            checkpoints=checkpoints,
            model_id="0",
            cache_dir="tmp_tracincpfast",
            checkpoints_load_func=load_state_dict,
            loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
            final_fc_layer=list(lit_model.model.children())[-1],
            device="cuda:0",
            batch_size=64,
        )

        method_save_dir = os.path.join(output_dir, "expl_tracincpfast")
        os.makedirs(method_save_dir, exist_ok=True)

        for i, (test_tensor, test_labels) in enumerate(test_dataloader):
            explanation_targets = [
                lit_model.model(test_tensor[i].unsqueeze(0).to("cuda:0")).argmax().item() for i in range(len(test_tensor))
            ]
            explanations_tracincpfast = explainer_tracincpfast.explain(test_tensor, targets=explanation_targets)
            EC.save(method_save_dir, explanations_tracincpfast, i)

    if method == "arnoldi":
        train_dataset = train_dataloader.dataset
        num_samples = 1000
        indices = generator.sample(range(len(train_dataset)), num_samples)
        hessian_dataset = Subset(train_dataset, indices)

        # Initialize Explainer
        explainer_arnoldi = CaptumArnoldi(
            model=lit_model,
            train_dataset=train_dataloader.dataset,
            hessian_dataset=hessian_dataset,
            checkpoint=checkpoints[0],
            loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
            checkpoints_load_func=load_state_dict,
            projection_dim=10,
            arnoldi_dim=200,
            layers=["model.fc"],  # only the last layer
            device="cuda:0",
        )

        method_save_dir = os.path.join(output_dir, "expl_arnoldi")
        os.makedirs(method_save_dir, exist_ok=True)
        # Explain test samples
        logger.info("Explaining test samples")
        for i, (test_tensor, test_labels) in enumerate(test_dataloader):
            explanation_targets = [
                lit_model.model(test_tensor[i].unsqueeze(0).to("cuda:0")).argmax().item() for i in range(len(test_tensor))
            ]
            explanations_arnoldi = explainer_arnoldi.explain(test=test_tensor, targets=explanation_targets)
            EC.save(method_save_dir, explanations_arnoldi, i)

    if method == "trak":
        explainer_trak = TRAK(
            model=lit_model.model,
            model_id="test_model",
            cache_dir="tmp_trak",
            train_dataset=train_dataloader.dataset,
            proj_dim=4096,
            load_from_disk=False,
        )

        method_save_dir = os.path.join(output_dir, "expl_trak")
        os.makedirs(method_save_dir, exist_ok=True)
        # Explain test samples
        for i, (test_tensor, test_labels) in enumerate(test_dataloader):
            explanation_targets = [
                lit_model.model(test_tensor[i].unsqueeze(0).to("cuda:0")).argmax().item() for i in range(len(test_tensor))
            ]
            explanations_trak = explainer_trak.explain(test=test_tensor, targets=explanation_targets)
            EC.save(method_save_dir, explanations_trak, i)


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
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to save outputs")
    parser.add_argument("--checkpoints_dir", required=True, type=str, help="Directory to checkpoints")
    parser.add_argument("--metadata_dir", required=True, type=str, help="Directory to metadata")
    parser.add_argument("--download", action="store_true", help="Download the datasets and checkpoints")
    args = parser.parse_args()

    # Call the function with parsed arguments
    compute_explanations(
        args.method,
        args.tiny_in_path,
        args.panda_sketch_path,
        args.output_dir,
        args.checkpoints_dir,
        args.metadata_dir,
        args.download,
    )
