import logging
import os
import random
import subprocess
from argparse import ArgumentParser

import lightning as L
import torch
import torchvision.transforms as transforms
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
from quanda.utils.cache import ExplanationsCache as EC
from quanda.utils.datasets.transformed import (
    LabelGroupingDataset,
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


def compute_explanations(method, tiny_in_path, panda_sketch_path, output_dir, checkpoints_dir, metadata_dir, download):
    torch.set_float32_matmul_precision("medium")

    # Downloading the datasets and checkpoints

    # We first download the datasets (uncomment the following cell if you haven't downloaded the datasets yet).:
    os.makedirs(output_dir, exist_ok=True)

    if download:
        os.makedirs(metadata_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)
        # os.makedirs(tiny_in_path, exist_ok=True)

        # subprocess.run(["wget", "-qP", tiny_in_path, "http://cs231n.stanford.edu/tiny-imagenet-200.zip"])
        # subprocess.run(["unzip", "-qq", os.path.join(tiny_in_path, "tiny-imagenet-200.zip"), "-d", tiny_in_path])
        subprocess.run(
            ["wget", "-qP", metadata_dir, "https://tinyurl.com/5chcwrbx"]
        )
        subprocess.run(["unzip", "-qq", os.path.join(metadata_dir, "sketch.zip"), "-d", metadata_dir])

        # Next we download all the necessary checkpoints and the dataset metadata
        subprocess.run(
            [
                "wget",
                "-qP",
                checkpoints_dir,
                "https://tinyurl.com/47tc84fu",
            ]
        )
        subprocess.run(["unzip", "-qq", "-j", os.path.join(checkpoints_dir, "tiny_inet_resnet18.zip"), "-d", metadata_dir])
        subprocess.run(
            ["wget", "-qP", metadata_dir, "https://tinyurl.com/u4w2j22k"]
        )
        subprocess.run(["unzip", "-qq", "-j", os.path.join(metadata_dir, "dataset_indices.zip"), "-d", metadata_dir])

    n_epochs = 10
    checkpoints = [
        os.path.join(checkpoints_dir, f"tiny_imagenet_resnet18_epoch={epoch:02d}.ckpt") for epoch in range(5, n_epochs, 1)
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
    device = "cuda:1"

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
    tiny_in_path = os.path.join(tiny_in_path, "tiny-imagenet-200/")
    with open(tiny_in_path + "wnids.txt", "r") as f:
        id_dict = {line.strip(): i for i, line in enumerate(f)}

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
        checkpoints = torch.load(path, map_location=torch.device("cuda:1"))
        module.model.load_state_dict(checkpoints["model_state_dict"])
        module.eval()
        return module.lr

    # Select test

    random_rng = random.Random(27)

    # get all cat classes
    cat_classes = [class_to_group[s] for s in class_to_group if class_to_group[s] == new_n_classes - 1]
    dog_classes = [class_to_group[s] for s in class_to_group if class_to_group[s] == new_n_classes - 2]

    all_cats = [s for s in range(len(test_set_grouped)) if test_set_grouped[s][1] in cat_classes]
    all_dogs = [s for s in range(len(test_set_grouped)) if test_set_grouped[s][1] in dog_classes]

    # find some correctly predicted cats and dogs
    correct_pred_cats = [
        i
        for i in all_cats
        if lit_model.model(test_set_grouped[i][0].unsqueeze(0).to(device)).argmax().item() == test_set_grouped[i][1]
    ]
    correct_pred_dogs = [
        i
        for i in all_dogs
        if lit_model.model(test_set_grouped[i][0].unsqueeze(0).to(device)).argmax().item() == test_set_grouped[i][1]
    ]
    test_dogs = random_rng.sample(correct_pred_dogs, 32)
    test_cats = random_rng.sample(correct_pred_cats, 32)

    # find regular samples
    all_clean_samples = [i for i in range(len(test_set_grouped)) if i not in all_cats + all_dogs]
    clean_samples = random_rng.sample(all_clean_samples, 64)
    test_shortcut = random_rng.sample(all_clean_samples, 64)

    mispredicted_clean = [
        i
        for i in all_clean_samples
        if lit_model.model(test_set_grouped[i][0].unsqueeze(0).to(device)).argmax().item() != test_set_grouped[i][1]
    ]
    test_mispredicted = random_rng.sample(mispredicted_clean, 64)

    correct_predict_panda = [
        i
        for i in range(len(all_panda))
        if lit_model(all_panda[i][0].unsqueeze(0).to(device)).argmax().item() == all_panda[i][1]
    ]
    torch.save(test_mispredicted, os.path.join(metadata_dir, "big_eval_test_mispredicted_indices.pth"))
    torch.save(test_shortcut, os.path.join(metadata_dir, "big_eval_test_shortcut_indices.pth"))
    torch.save(test_dogs, os.path.join(metadata_dir, "big_eval_test_dogs_indices.pth"))
    torch.save(test_cats, os.path.join(metadata_dir, "big_eval_test_cats_indices.pth"))
    torch.save(clean_samples, os.path.join(metadata_dir, "big_eval_test_clean_indices.pth"))
    torch.save(correct_predict_panda, os.path.join(metadata_dir, "big_eval_test_correct_predict_panda_indices.pth"))

    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    def vis_dataloader(dataloader):
        images, labels = next(iter(dataloader))
        images = denormalize(images)
        fig, ax = plt.subplots(figsize=(12, 6))
        plt.title("Sample images from CIFAR10 dataset")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()

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
    dataloaders["subclass"] = torch.utils.data.DataLoader(
        cat_dog_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    # vis_dataloader(dataloaders["subclass"])

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

    if method == "similarity":
        # Initialize Explainer
        explainer_similarity = CaptumSimilarity(
            model=lit_model,
            model_id="0",
            cache_dir=output_dir,
            train_dataset=train_dataloader.dataset,
            layers="model.avgpool",
            similarity_metric=cosine_similarity,
            device=device,
            batch_size=batch_size,
            load_from_disk=False,
        )

        method_save_dir = os.path.join(output_dir, method)
        os.makedirs(method_save_dir, exist_ok=True)
        # Explain test samples
        logger.info("Explaining test samples")
        for subset in dataloaders:
            for i, (test_tensor, test_labels) in enumerate(dataloaders[subset]):
                subset_save_dir = os.path.join(method_save_dir, subset)
                os.makedirs(subset_save_dir, exist_ok=True)
                explanations_similarity = explainer_similarity.explain(test_tensor)
                EC.save(subset_save_dir, explanations_similarity, i)

    if method == "representer_points":
        explainer_repr = RepresenterPoints(
            model=lit_model,
            cache_dir=output_dir,
            train_dataset=train_dataloader.dataset,
            features_layer="model.avgpool",
            classifier_layer="model.fc",
            batch_size=batch_size,
            features_postprocess=lambda x: x[:, :, 0, 0],
            model_id="demo",
            load_from_disk=False,
            show_progress=False,
        )

        method_save_dir = os.path.join(output_dir, method)
        os.makedirs(method_save_dir, exist_ok=True)
        # Explain test samples
        logger.info("Explaining test samples")
        for subset in dataloaders:
            for i, (test_tensor, test_labels) in enumerate(dataloaders[subset]):
                subset_save_dir = os.path.join(method_save_dir, subset)
                os.makedirs(subset_save_dir, exist_ok=True)
                explanation_targets = lit_model.model(test_tensor.to(device)).argmax(dim=1)
                explanations_repr = explainer_repr.explain(test_tensor, explanation_targets)
                EC.save(subset_save_dir, explanations_repr, i)

    if method == "tracincpfast":
        # Initialize Explainer
        logger.info("Explaining test samples")
        explainer_tracincpfast = CaptumTracInCPFast(
            model=lit_model,
            train_dataset=train_dataloader.dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=load_state_dict,
            loss_fn=torch.nn.CrossEntropyLoss(reduction="mean"),
            final_fc_layer=list(lit_model.model.children())[-1],
            device=device,
            batch_size=batch_size * 4,
        )

        method_save_dir = os.path.join(output_dir, method)
        os.makedirs(method_save_dir, exist_ok=True)

        for subset in dataloaders:
            for i, (test_tensor, test_labels) in enumerate(dataloaders[subset]):
                subset_save_dir = os.path.join(method_save_dir, subset)
                os.makedirs(subset_save_dir, exist_ok=True)
                explanation_targets = lit_model.model(test_tensor.to(device)).argmax(dim=1)
                import time

                # Explain test samples
                start_time = time.time()
                explanations_tracincpfast = explainer_tracincpfast.explain(test_tensor, targets=explanation_targets)
                end_time = time.time()
                EC.save(subset_save_dir, explanations_tracincpfast, i)
                print(f"Time taken for subset {subset} is {end_time - start_time}")

    if method == "arnoldi":
        train_dataset = train_dataloader.dataset
        num_samples = 5000
        indices = generator.sample(range(len(train_dataset)), num_samples)
        hessian_dataset = Subset(train_dataset, indices)

        # Initialize Explainer
        explainer_arnoldi = CaptumArnoldi(
            model=lit_model,
            train_dataset=train_dataloader.dataset,
            hessian_dataset=hessian_dataset,
            checkpoint=checkpoints[-1],
            loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
            checkpoints_load_func=load_state_dict,
            projection_dim=500,
            arnoldi_dim=200,
            batch_size=batch_size * 4,
            #layers=["model.fc"],  # only the last layer
            device=device,
        )

        method_save_dir = os.path.join(output_dir, method)
        os.makedirs(method_save_dir, exist_ok=True)
        # Explain test samples
        logger.info("Explaining test samples")
        for subset in dataloaders:
            for i, (test_tensor, test_labels) in enumerate(dataloaders[subset]):
                subset_save_dir = os.path.join(method_save_dir, subset)
                os.makedirs(subset_save_dir, exist_ok=True)
                explanation_targets = lit_model.model(test_tensor.to(device)).argmax(dim=1)
                explanations_arnoldi = explainer_arnoldi.explain(test=test_tensor, targets=explanation_targets)
                EC.save(subset_save_dir, explanations_arnoldi, i)

    if method == "trak":
        explainer_trak = TRAK(
            model=lit_model.model,
            model_id="test_model",
            cache_dir=output_dir,
            train_dataset=train_dataloader.dataset,
            projector="cuda",
            proj_dim=4096,
            batch_size=batch_size,
            load_from_disk=False,
        )

        method_save_dir = os.path.join(output_dir, method)
        os.makedirs(method_save_dir, exist_ok=True)
        # Explain test samples
        for subset in dataloaders:
            for i, (test_tensor, test_labels) in enumerate(dataloaders[subset]):
                subset_save_dir = os.path.join(method_save_dir, subset)
                os.makedirs(subset_save_dir, exist_ok=True)
                explanation_targets = lit_model.model(test_tensor.to(device)).argmax(dim=1)
                explanations_trak = explainer_trak.explain(test=test_tensor, targets=explanation_targets)
                EC.save(subset_save_dir, explanations_trak, i)

    if method == "random":
        explainer_rand = RandomExplainer(
            model=lit_model,
            train_dataset=train_dataloader.dataset,
            seed=27,
        )

        method_save_dir = os.path.join(output_dir, method)
        os.makedirs(method_save_dir, exist_ok=True)
        # Explain test samples
        for subset in dataloaders:
            for i, (test_tensor, test_labels) in enumerate(dataloaders[subset]):
                subset_save_dir = os.path.join(method_save_dir, subset)
                os.makedirs(subset_save_dir, exist_ok=True)
                explanation_targets = lit_model.model(test_tensor.to(device)).argmax(dim=1)
                explanations_rand = explainer_rand.explain(test=test_tensor, targets=explanation_targets)
                EC.save(subset_save_dir, explanations_rand, i)


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
