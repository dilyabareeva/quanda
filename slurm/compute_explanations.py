import os
import random
import subprocess
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import pytorch_lightning as pl
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
from quanda.utils.datasets.transformed import LabelGroupingDataset
from quanda.utils.functions import cosine_similarity
from tutorials.utils.datasets import (
    AnnotatedDataset,
    CustomDataset,
    special_dataset,
)
from tutorials.utils.modules import LitModel
from tutorials.utils.visualization import visualize_top_3_bottom_3_influential


def compute_explanations(method, tiny_in_path, panda_sketch_path, save_dir):
    torch.set_float32_matmul_precision("medium")

    # Downloading the datasets and checkpoints

    # We first download the datasets (uncomment the following cell if you haven't downloaded the datasets yet).:
    os.makedirs(save_dir, exist_ok=True)

    subprocess.run(["wget", "-P", tiny_in_path, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])
    subprocess.run(["unzip", os.path.join(tiny_in_path, "tiny-imagenet-200.zip"), "-d", tiny_in_path])
    subprocess.run(["wget", "-P", save_dir, "https://datacloud.hhi.fraunhofer.de/s/FpPWkzPmM3s9ZqF/download/sketch.zip"])
    subprocess.run(["unzip", os.path.join(save_dir, "sketch.zip"), "-d", save_dir])

    # Next we download all the necessary checkpoints and the dataset metadata (uncomment the following cell if you haven't downloaded the checkpoints yet).:
    subprocess.run(
        ["wget", "-P", save_dir, "https://datacloud.hhi.fraunhofer.de/s/ZE5dBnfzW94Xkoo/download/tiny_inet_resnet18.zip"]
    )
    subprocess.run(["unzip", "-j", os.path.join(save_dir, "tiny_inet_resnet18.zip"), "-d", save_dir])
    subprocess.run(
        ["wget", "-P", save_dir, "https://datacloud.hhi.fraunhofer.de/s/AmnCXAC8zx3YQgP/download/dataset_indices.zip"]
    )
    subprocess.run(["unzip", "-j", os.path.join(save_dir, "dataset_indices.zip"), "-d", save_dir])

    n_epochs = 10
    checkpoints = [os.path.join(save_dir, f"tiny_imagenet_resnet18_epoch={epoch:02d}.ckpt") for epoch in range(1, n_epochs, 2)]

    # Dataset Construction

    # Loading the dataset metadata
    class_to_group = torch.load(os.path.join(save_dir, "class_to_group.pth"))
    r_name_dict = torch.load(os.path.join(save_dir, "r_name_dict.pth"))
    test_indices = torch.load(os.path.join(save_dir, "main_test_indices.pth"))
    test_split = torch.load(os.path.join(save_dir, "test_indices.pth"))
    val_split = torch.load(os.path.join(save_dir, "val_indices.pth"))

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

    denormalize = transforms.Compose(
        [transforms.Normalize(mean=[0, 0, 0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])]
        + [transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1])]
    )

    # Load the TinyImageNet dataset
    id_dict = {}
    with open(tiny_in_path + "/tiny-imagenet-200/wnids.txt", "r") as f:
        id_dict = {line.strip(): i for i, line in enumerate(f)}

    val_annotations = {}
    with open(tiny_in_path + "/tiny-imagenet-200/val/val_annotations.txt", "r") as f:
        val_annotations = {line.split("\t")[0]: line.split("\t")[1] for line in f}

    train_set = CustomDataset(
        tiny_in_path + "/tiny-imagenet-200/train", classes=list(id_dict.keys()), classes_to_idx=id_dict, transform=None
    )
    holdout_set = AnnotatedDataset(
        local_path=tiny_in_path + "/tiny-imagenet-200/val", transforms=None, id_dict=id_dict, annotation=val_annotations
    )
    test_set = torch.utils.data.Subset(holdout_set, test_split)
    val_set = torch.utils.data.Subset(holdout_set, val_split)

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
        pomegranate_class=162,
        p_shortcut=0.4,
        p_flipping=0.1,
        dog_class=189,
        cat_class=190,
        shortcut_transform_indices=torch.load(os.path.join(save_dir, "all_train_shortcut_indices_for_generation.pth")),
        flipping_transform_dict=torch.load(os.path.join(save_dir, "all_train_flipped_dict_for_generation.pth")),
    )

    test_set_clean = LabelGroupingDataset(
        dataset=test_set,
        n_classes=n_classes,
        dataset_transform=regular_transforms,
        class_to_group=class_to_group,
    )

    test_set = special_dataset(
        test_set,
        n_classes,
        new_n_classes,
        regular_transforms,
        class_to_group=class_to_group,
        shortcut_fn=add_yellow_square,
        backdoor_dataset=panda_test,
        pomegranate_class=None,
        p_shortcut=0.3,
        p_flipping=0.1,
        dog_class=189,
        cat_class=190,
        shortcut_transform_indices=torch.load(os.path.join(save_dir, "all_test_shortcut_indices_for_generation.pth")),
        flipping_transform_dict={},
    )

    random_rng = random.Random(27)

    all_backdoor = torch.load(os.path.join(save_dir, "all_test_backdoor_indices.pth"))
    test_backd = random_rng.sample(all_backdoor, 16)
    all_shortcut = torch.load(os.path.join(save_dir, "all_test_shortcut_indices.pth"))
    act_test_sc_non_pom = [s for s in test_backd if test_set[s][1] != 162]
    test_shortc = random_rng.sample(act_test_sc_non_pom, 16)

    all_cats = [s for s in range(len(test_set)) if test_set[s][1] in [new_n_classes - 1]]
    all_dogs = [s for s in range(len(test_set)) if test_set[s][1] in [new_n_classes - 2]]
    test_dogs_cats = random_rng.sample(all_cats, 16)
    test_dogs_cats += random_rng.sample(all_dogs, 16)

    all_clean_samples = [i for i in range(len(test_set)) if i not in all_backdoor + all_shortcut + test_dogs_cats]
    clean_samples = random_rng.sample(all_clean_samples, 16)

    # backdoor, shortcut, dogs and cats samples
    test_indices = test_backd + test_shortc + test_dogs_cats + clean_samples
    target_test_set = torch.utils.data.Subset(test_set, test_indices)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(
        target_test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

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

        method_save_dir = os.path.join(save_dir, "similarity")
        # Explain test samples
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
            train_labels=torch.tensor(explanation_targets),
            features_layer="model.avgpool",
            classifier_layer="model.fc",
            batch_size=32,
            features_postprocess=lambda x: x[:, :, 0, 0],
            model_id="demo",
            load_from_disk=False,
            show_progress=False,
        )

        method_save_dir = os.path.join(save_dir, "representer_points")
        # Explain test samples
        for i, (test_tensor, test_labels) in enumerate(test_dataloader):
            explanation_targets = [
                lit_model.model(test_tensor[i].unsqueeze(0).to("cuda:0")).argmax().item() for i in range(len(test_tensor))
            ]
            explanations_repr = explainer_repr.explain(test_tensor, explanation_targets)
            EC.save(method_save_dir, explanations_repr, i)

    if method == "tracincpfast":

        def load_state_dict(module: pl.LightningModule, path: str) -> int:
            module = type(module).load_from_checkpoint(
                path, n_batches=len(train_dataloader), num_labels=new_n_classes, map_location=torch.device("cuda:0")
            )
            module.model.eval()
            return module.lr

        # Initialize Explainer
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

        method_save_dir = os.path.join(save_dir, "tracincpfast")

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

        method_save_dir = os.path.join(save_dir, "arnoldi")
        # Explain test samples
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

        method_save_dir = os.path.join(save_dir, "trak")
        # Explain test samples
        for i, (test_tensor, test_labels) in enumerate(test_dataloader):
            explanation_targets = [
                lit_model.model(test_tensor[i].unsqueeze(0).to("cuda:0")).argmax().item() for i in range(len(test_tensor))
            ]
            explanations_trak = explainer_arnoldi.explain(test=test_tensor, targets=explanation_targets)
            EC.save(method_save_dir, explanations_trak, i)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--method", required=True, type=int)
    args = parser.parse_args()
    main(args.model_id)
