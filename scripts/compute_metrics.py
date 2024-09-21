import logging
import os
import random
import subprocess
from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
import wandb
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Subset

from quanda.metrics.downstream_eval import (
    MislabelingDetectionMetric,
    ShortcutDetectionMetric,
    SubclassDetectionMetric,
)
from quanda.metrics.heuristics import MixedDatasetsMetric, TopKOverlapMetric
from quanda.utils.cache import ExplanationsCache as EC
from quanda.utils.datasets.transformed import (
    LabelGroupingDataset,
    TransformedDataset,
)
from tutorials.utils.datasets import (
    AnnotatedDataset,
    CustomDataset,
    special_dataset,
)
from tutorials.utils.modules import LitModel

logger = logging.getLogger(__name__)


load_dotenv()


def compute_metrics(metric, tiny_in_path, panda_sketch_path, explanations_dir, checkpoints_dir, metadata_dir, download):
    torch.set_float32_matmul_precision("medium")

    # Downloading the datasets and checkpoints

    # Initialize WandbLogger
    wandb.init(project="quanda", name="tiny_inet_resnet18")

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

    # Define transformations
    regular_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    )

    # Load the TinyImageNet dataset
    tiny_in_path = os.path.join(tiny_in_path, "tiny-imagenet-200/")
    with open(tiny_in_path + "wnids.txt", "r") as f:
        id_dict = {line.strip(): i for i, line in enumerate(f)}

    with open(tiny_in_path + "val/val_annotations.txt", "r") as f:
        val_annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}

    train_set_raw = CustomDataset(tiny_in_path + "train", classes=list(id_dict.keys()), classes_to_idx=id_dict, transform=None)
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

    backdoor_transforms_flipped = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(1.0),
        ]
    )

    panda_dataset = CustomDataset(
        panda_sketch_path, classes=["n02510455"], classes_to_idx={"n02510455": 5}, transform=backdoor_transforms
    )
    panda_twin_dataset = CustomDataset(
        panda_sketch_path, classes=["n02510455"], classes_to_idx={"n02510455": 5}, transform=backdoor_transforms_flipped
    )

    panda_set = torch.utils.data.Subset(panda_dataset, panda_train_indices)
    panda_rest_indices = [i for i in range(len(panda_dataset)) if i not in panda_train_indices]
    panda_test = torch.utils.data.Subset(panda_dataset, panda_rest_indices)
    panda_twin = torch.utils.data.Subset(panda_twin_dataset, panda_rest_indices)
    all_panda = torch.utils.data.ConcatDataset([panda_test, panda_twin])


    def add_yellow_square(img):
        square_size = (15, 15)  # Size of the square
        yellow_square = Image.new("RGB", square_size, (255, 255, 0))  # Create a yellow square
        img.paste(yellow_square, (10, 10))  # Paste it onto the image at the specified position
        return img

    train_set = special_dataset(
        train_set_raw,
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

    # add regular_transforms to test_set
    test_set_transform = TransformedDataset(
        dataset=test_set,
        n_classes=new_n_classes,
        dataset_transform=regular_transforms,
        transform_indices=[],
    )

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    lit_model = LitModel.load_from_checkpoint(
        checkpoints[-1], n_batches=len(train_dataloader), num_labels=new_n_classes, device=device, map_location=torch.device(device)
    )
    lit_model.to(device)
    lit_model.model = lit_model.model.eval()

    # Define Dataloader for different metrics
    dataloaders = {}
    # Dataloader for Mislabeling Detection
    test_mispredicted = torch.load(os.path.join(metadata_dir, "big_eval_test_mispredicted_indices.pth"))
    mispredicted_dataset = torch.utils.data.Subset(test_set_grouped, test_mispredicted)

    dataloaders["mislabeling"] = torch.utils.data.DataLoader(
        mispredicted_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # vis_dataloader(dataloaders["mislabeling"])

    # Dataloder for Shortcut Detection
    test_shortcut = torch.load(os.path.join(metadata_dir, "big_eval_test_shortcut_indices.pth"))
    shortcut_dataset = TransformedDataset(
        dataset=torch.utils.data.Subset(test_set, test_shortcut),
        n_classes=new_n_classes,
        dataset_transform=regular_transforms,
        transform_indices=list(range(len(test_shortcut))),
        sample_fn=add_yellow_square,
        label_fn=lambda x: class_to_group[x],
    )
    dataloaders["shortcut"] = torch.utils.data.DataLoader(
        shortcut_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # vis_dataloader(dataloaders["shortcut"])

    # Dataloader for subclass detection
    test_dogs = torch.load(os.path.join(metadata_dir, "big_eval_test_dogs_indices.pth"))
    test_cats = torch.load(os.path.join(metadata_dir, "big_eval_test_cats_indices.pth"))
    cat_dog_dataset = torch.utils.data.Subset(test_set_grouped, test_cats + test_dogs)
    cat_dog_ungrouped_dataset = torch.utils.data.Subset(test_set_transform, test_cats + test_dogs)
    dataloaders["subclass"] = torch.utils.data.DataLoader(
        cat_dog_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    dataloaders["subclass_ungrouped"] = torch.utils.data.DataLoader(
        cat_dog_ungrouped_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # vis_dataloader(dataloaders["cat_dog"])

    # Dataloader for Model Randomization, Top-K Overlap
    clean_samples = torch.load(os.path.join(metadata_dir, "big_eval_test_clean_indices.pth"))
    clean_dataset = torch.utils.data.Subset(test_set_grouped, clean_samples)
    dataloaders["randomization"] = torch.utils.data.DataLoader(
        clean_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    dataloaders["top_k_overlap"] = dataloaders["randomization"]
    # vis_dataloader(dataloaders["randomization"])

    # Dataloader for Mixed Datasets
    correct_predict_panda = torch.load(os.path.join(metadata_dir, "big_eval_test_correct_predict_panda_indices.pth"))
    dataloaders["mixed_dataset"] = torch.utils.data.DataLoader(
        torch.utils.data.Subset(all_panda, correct_predict_panda),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    # vis_dataloader(dataloaders["mixed_dataset"])


    explanation_methods = ["similarity", "representer_points", "trak", "random", "tracincpfast", "arnoldi"]
    if metric == "mislabeling":
        for method in explanation_methods:
            method_save_dir = os.path.join(explanations_dir, method)
            subset_save_dir = os.path.join(method_save_dir, metric)
            explanations = EC.load(subset_save_dir)
            mislabeled = MislabelingDetectionMetric(
                model=lit_model,
                train_dataset=train_set,
                mislabeling_indices=torch.load(os.path.join(metadata_dir, "all_train_flipped_indices.pth")),
                global_method="sum_abs",
            )
            for i, (test_tensor, test_labels) in enumerate(dataloaders[metric]):
                test_tensor, test_labels = test_tensor.to(device), test_labels.to(device)
                mislabeled.update(test_tensor, test_labels, explanations[i].to(device))

            score = mislabeled.compute()
            wandb.log({f"{method}_{metric}": score})

    if metric == "shortcut":
        for method in explanation_methods:
            method_save_dir = os.path.join(explanations_dir, method)
            subset_save_dir = os.path.join(method_save_dir, metric)
            explanations = EC.load(subset_save_dir)
            shortcut = ShortcutDetectionMetric(
                model=lit_model,
                train_dataset=train_set,
                shortcut_indices=torch.load(os.path.join(metadata_dir, "all_train_shortcut_indices.pth")),
                shortcut_cls=162,
                filter_by_prediction=False,
                filter_by_class=False,
            )
            for i, (test_tensor, test_labels) in enumerate(dataloaders[metric]):
                test_tensor, test_labels = test_tensor.to(device), test_labels.to(device)
                explanation_targets = [
                    lit_model.model(test_tensor[i].unsqueeze(0).to(device)).argmax().item() for i in range(len(test_tensor))
                ]
                shortcut.update(explanations[i].to(device))

            score = shortcut.compute()
            wandb.log({f"{method}_{metric}": score})

    if metric == "subclass":
        train_subclass = torch.tensor([5 for s in panda_set] + [s[1] for s in train_set_raw])

        for method in explanation_methods:
            ungrouped_iter = iter(dataloaders["subclass_ungrouped"])
            method_save_dir = os.path.join(explanations_dir, method)
            subset_save_dir = os.path.join(method_save_dir, metric)
            explanations = EC.load(subset_save_dir)
            id_subclass = SubclassDetectionMetric(
                model=lit_model,
                train_dataset=train_set,
                train_subclass_labels=train_subclass,
            )
            for i, (test_tensor, test_labels) in enumerate(dataloaders[metric]):
                test_sublabels = next(ungrouped_iter)[1]
                test_tensor, test_labels = test_tensor.to(device), test_labels.to(device)
                explanation_targets = [
                    lit_model.model(test_tensor[i].unsqueeze(0).to(device)).argmax().item() for i in range(len(test_tensor))
                ]
                id_subclass.update(test_sublabels, explanations[i])

            score = id_subclass.compute()
            wandb.log({f"{method}_{metric}": score})

    if metric == "top_k_overlap":
        for method in explanation_methods:
            method_save_dir = os.path.join(explanations_dir, method)
            subset_save_dir = os.path.join(method_save_dir, metric)
            explanations = EC.load(subset_save_dir)
            top_k = TopKOverlapMetric(model=lit_model, train_dataset=train_set, top_k=1)
            for i, (test_tensor, test_labels) in enumerate(dataloaders[metric]):
                test_tensor, test_labels = test_tensor.to(device), test_labels.to(device)
                explanation_targets = [
                    lit_model.model(test_tensor[i].unsqueeze(0).to(device)).argmax().item() for i in range(len(test_tensor))
                ]
                top_k.update(explanations[i].to(device))

            score = top_k.compute()
            wandb.log({f"{method}_{metric}": score})

    if metric == "mixed_dataset":
        all_adv_indices = torch.load(os.path.join(metadata_dir, "all_train_backdoor_indices.pth"))
        # to binary
        adv_indices = torch.tensor([1 if i in all_adv_indices else 0 for i in range(len(train_set))])

        for method in explanation_methods:
            method_save_dir = os.path.join(explanations_dir, method)
            subset_save_dir = os.path.join(method_save_dir, metric)
            explanations = EC.load(subset_save_dir)
            mixed_dataset = MixedDatasetsMetric(
                train_dataset=train_set,
                model=lit_model,
                adversarial_indices=adv_indices,
            )
            for i, (test_tensor, test_labels) in enumerate(dataloaders[metric]):
                test_tensor, test_labels = test_tensor.to(device), test_labels.to(device)
                explanation_targets = [
                    lit_model.model(test_tensor[i].unsqueeze(0).to(device)).argmax().item() for i in range(len(test_tensor))
                ]
                mixed_dataset.update(explanations[i].to(device))

            score = mixed_dataset.compute()
            wandb.log({f"{method}_{metric}": score})


if __name__ == "__main__":
    parser = ArgumentParser()

    # Define argument for metric with choices
    parser.add_argument(
        "--metric",
        required=True,
        choices=["mislabeling", "shortcut", "subclass", "top_k_overlap", "mixed_dataset"],
        help="Choose the explanation metric to use.",
    )

    # Define other required arguments
    parser.add_argument("--tiny_in_path", required=True, type=str, help="Path to Tiny ImageNet dataset")
    parser.add_argument("--panda_sketch_path", required=True, type=str, help="Path to ImageNet-Sketch dataset")
    parser.add_argument("--explanations_dir", required=True, type=str, help="Directory where explanations are stored")
    parser.add_argument("--checkpoints_dir", required=True, type=str, help="Directory to checkpoints")
    parser.add_argument("--metadata_dir", required=True, type=str, help="Directory to metadata")
    parser.add_argument("--download", action="store_true", help="Download the datasets and checkpoints")
    args = parser.parse_args()

    # Call the function with parsed arguments
    compute_metrics(
        args.metric,
        args.tiny_in_path,
        args.panda_sketch_path,
        args.explanations_dir,
        args.checkpoints_dir,
        args.metadata_dir,
        args.download,
    )
