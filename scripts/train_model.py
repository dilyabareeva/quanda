import logging
import os
import sys

from typing import Optional, Callable

sys.path.append(os.getcwd())

from argparse import ArgumentParser

import torch
import torchvision.transforms as transforms
from torch.utils.data import Subset
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
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from quanda.utils.datasets.transformed import (
    LabelFlippingDataset,
    SampleTransformationDataset,
    LabelGroupingDataset,
)
from quanda.utils.datasets import SingleClassImageDataset

from quanda.benchmarks import Benchmark
from quanda.benchmarks.heuristics import MixedDatasets
from quanda.benchmarks.resources.sample_transforms import sample_transforms
from quanda.benchmarks.resources.modules import (
    load_module_with_name,
)


logger = logging.getLogger(__name__)

datasets_metadata = {
    "mnist": {
        "hf_tag": "ylecun/mnist",
        "validation_size": 3000,
        "test_split_name": "test",
        "num_classes": 10,
        "shortcut_cls": 5,
        "num_groups": 2,
        "shortcut_probability": 0.3,
        "mislabeling_probability": 0.2,
        "adversarial_cls": 0,
        "adversarial_dir_url": "https://datacloud.hhi.fraunhofer.de/s/LAzkbk9mm6L3Lz7/download/fasion_mnist_150.zip",
    },
    "tiny_imagenet": {
        "hf_tag": "zh-plus/tiny-imagenet-200",
        "validation_size": 3000,
        "test_split_name": "val",
        "num_classes": 200,
        "shortcut_cls": 100,
        "num_groups": 2,
        "shortcut_probability": 0.3,
        "mislabeling_probability": 0.2,
        "adversarial_cls": 0,
        "adversarial_dir_url": None,
    },
}


def load_augmentation(name: str, dataset_name: str) -> Callable:
    if name is None or name == "null":
        return lambda x: x
    shapes = {"tiny_imagenet": (64, 64), "mnist": (28, 28)}
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


def handle_mislabeled_dataset(
    train_set: torch.utils.data.Dataset,
    val_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    metadata_path: str,
    dataset_name: str,
    output_path: str,
    num_classes: int,
    seed: int,
    regular_transforms: Callable,
    mislabeling_probability: float,
):
    if not os.path.exists(os.path.join(metadata_path, "mislabeling_indices")):
        assert mislabeling_probability is not None, (
            "mislabeling_probability must be given to create mislabeled dataset from scratch"
        )
        train_set = LabelFlippingDataset(
            dataset=train_set,
            n_classes=num_classes,
            dataset_transform=regular_transforms,
            p=mislabeling_probability,
            seed=seed,
        )
        mislabeling_indices = train_set.transform_indices
        torch.save(
            mislabeling_indices,
            os.path.join(output_path, "mislabeling_indices"),
        )
        mislabeling_labels = torch.tensor(
            [train_set.mislabeling_labels[i] for i in mislabeling_indices]
        )
        torch.save(
            mislabeling_labels,
            os.path.join(output_path, "mislabeling_labels"),
        )
    else:
        mislabeling_indices = torch.load(
            os.path.join(metadata_path, "mislabeling_indices")
        )
        mislabeling_labels = torch.load(
            os.path.join(metadata_path, "mislabeling_labels")
        )
        mislabeling_labels = {
            mislabeling_indices[i]: int(mislabeling_labels[i])
            for i in range(mislabeling_labels.shape[0])
        }
        train_set = LabelFlippingDataset(
            dataset=train_set,
            n_classes=num_classes,
            dataset_transform=regular_transforms,
            transform_indices=mislabeling_indices,
            mislabeling_labels=mislabeling_labels,
            seed=seed,
        )
    return_dict = {
        "mislabeling_indices": mislabeling_indices,
        "mislabeling_labels": mislabeling_labels,
    }
    return train_set, val_set, test_set, return_dict


def handle_mixed_dataset(
    train_set: torch.utils.data.Dataset,
    val_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    metadata_path: str,
    dataset_name: str,
    output_path: str,
    num_classes: int,
    seed: int,
    regular_transforms: Callable,
    adversarial_cls: int,
    adversarial_dir: str,
):
    assert adversarial_dir is not None, (
        "adversarial_dir must be given to create mixed dataset"
    )
    temp_benchmark = MixedDatasets()
    adversarial_dataset_path = os.path.join(
        adversarial_dir, "mixed_datasets_adversarial_dataset"
    )
    if not os.path.exists(
        os.path.join(adversarial_dir, "adversarial_dataset.zip")
    ):
        adversarial_dataset_path = (
            temp_benchmark._download_adversarial_dataset(
                adversarial_dir_url=datasets_metadata[dataset_name][
                    "adversarial_dir_url"
                ],
                cache_dir=adversarial_dir,
            )
        )
    if not os.path.exists(
        os.path.join(metadata_path, "adversarial_train_indices")
    ):
        temp_dataset = SingleClassImageDataset(
            root=adversarial_dataset_path,
            label=adversarial_cls,
            transform=None,
        )
        adversarial_size = int(
            len(temp_dataset) * 0.5
        )  # TODO: make this a parameter
        all_indices = torch.randperm(len(temp_dataset))
        adversarial_train_indices = all_indices[:adversarial_size]
        torch.save(
            adversarial_train_indices,
            os.path.join(output_path, "adversarial_train_indices"),
        )
        adversarial_test_indices = all_indices[adversarial_size:]
        torch.save(
            adversarial_test_indices,
            os.path.join(output_path, "adversarial_test_indices"),
        )
    else:
        adversarial_train_indices = torch.load(
            os.path.join(metadata_path, "adversarial_train_indices")
        )
        adversarial_test_indices = torch.load(
            os.path.join(metadata_path, "adversarial_test_indices")
        )
    adversarial_dataset = SingleClassImageDataset(
        root=adversarial_dataset_path,
        label=adversarial_cls,
        transform=sample_transforms[f"{dataset_name}_adversarial_transform"],
        indices=adversarial_train_indices,
    )
    # adversarial_indices = [1] * len(adversarial_dataset) + [0] * len(
    #    train_set
    # )
    train_set = torch.utils.data.ConcatDataset(
        [adversarial_dataset, train_set]
    )
    return_dict = {
        "adversarial_train_indices": adversarial_train_indices,
        "adversarial_test_indices": adversarial_test_indices,
        "adversarial_cls": adversarial_cls,
        "adversarial_dir_url": datasets_metadata[dataset_name][
            "adversarial_dir_url"
        ],
    }
    return train_set, val_set, test_set, return_dict


def handle_shortcut_dataset(
    train_set: torch.utils.data.Dataset,
    val_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    metadata_path: str,
    dataset_name: str,
    output_path: str,
    num_classes: int,
    seed: int,
    regular_transforms: Callable,
    shortcut_probability: float,
    shortcut_cls: int,
):
    if not os.path.exists(os.path.join(metadata_path, "shortcut_indices")):
        assert shortcut_probability is not None, (
            "shortcut_probability must be given to create shortcut dataset from scratch"
        )
        train_set = SampleTransformationDataset(
            dataset=train_set,
            n_classes=num_classes,
            sample_fn=sample_transforms[f"{dataset_name}_shortcut_transform"],
            dataset_transform=regular_transforms,
            cls_idx=shortcut_cls,
            p=shortcut_probability,
            seed=seed,
        )
        torch.save(
            train_set.transform_indices,
            os.path.join(output_path, "shortcut_indices"),
        )
        shortcut_indices = train_set.transform_indices
    else:
        shortcut_indices = torch.load(
            os.path.join(metadata_path, "shortcut_indices")
        )
        train_set = SampleTransformationDataset(
            dataset=train_set,
            n_classes=num_classes,
            sample_fn=sample_transforms[f"{dataset_name}_shortcut_transform"],
            dataset_transform=regular_transforms,
            transform_indices=shortcut_indices,
            seed=seed,
        )
    return_dict = {
        "shortcut_cls": shortcut_cls,
        "shortcut_indices": shortcut_indices,
    }
    return train_set, val_set, test_set, return_dict


def handle_subclass_dataset(
    train_set: torch.utils.data.Dataset,
    val_set: torch.utils.data.Dataset,
    test_set: torch.utils.data.Dataset,
    metadata_path: str,
    dataset_name: str,
    output_path: str,
    num_classes: int,
    seed: int,
    regular_transforms: Callable,
    num_groups: int,
):
    if not os.path.exists(os.path.join(metadata_path, "class_to_group")):
        train_set = LabelGroupingDataset(
            dataset=train_set,
            n_classes=num_classes,
            dataset_transform=None,
            seed=seed,
            n_groups=num_groups,
            class_to_group="random",
        )
        torch.save(
            train_set.class_to_group,
            os.path.join(output_path, "class_to_group"),
        )
    else:
        class_to_group = torch.load(
            os.path.join(metadata_path, "class_to_group")
        )
        train_set = LabelGroupingDataset(
            dataset=train_set,
            n_classes=num_classes,
            dataset_transform=None,
            seed=seed,
            class_to_group=class_to_group,
        )
    val_set = LabelGroupingDataset(
        dataset=val_set,
        n_classes=num_classes,
        dataset_transform=None,
        seed=seed,
        class_to_group=train_set.class_to_group,
    )
    return_dict = {"class_to_group": train_set.class_to_group}

    return train_set, val_set, test_set, return_dict


def load_datasets(
    dataset_name: str,
    dataset_cache_dir: str,
    augmentation: str,
    dataset_type: str,
    metadata_path: str,
    output_path: str,
    seed: int,
    adversarial_dir: Optional[str] = None,
):
    regular_transforms = sample_transforms[f"{dataset_name}_transform"]
    adversarial_cls = datasets_metadata[dataset_name]["adversarial_cls"]
    shortcut_cls = datasets_metadata[dataset_name]["shortcut_cls"]
    num_classes = datasets_metadata[dataset_name]["num_classes"]
    num_groups = datasets_metadata[dataset_name]["num_groups"]
    shortcut_probability = datasets_metadata[dataset_name][
        "shortcut_probability"
    ]
    mislabeling_probability = datasets_metadata[dataset_name][
        "mislabeling_probability"
    ]

    if augmentation is not None and augmentation != "":
        augmentation = load_augmentation(augmentation, dataset_name)

    augmented_transform = (
        transforms.Compose([augmentation, regular_transforms])
        if augmentation is not None
        else regular_transforms
    )
    train_set = Benchmark._process_dataset(
        Benchmark,
        dataset=datasets_metadata[dataset_name]["hf_tag"],
        dataset_split="train",
        transform=augmentation
        if dataset_type in ["mislabeled", "shortcut"]
        else augmented_transform,
        cache_dir=dataset_cache_dir,
    )

    # TODO: Currently, holdout sets are vanilla except when dataset_type=="subclass"
    holdout_set = Benchmark._process_dataset(
        Benchmark,
        dataset=datasets_metadata[dataset_name]["hf_tag"],
        dataset_split=datasets_metadata[dataset_name]["test_split_name"],
        transform=regular_transforms,
        cache_dir=dataset_cache_dir,
    )
    if os.path.exists(os.path.join(metadata_path, "validation_indices")):
        val_indices = torch.load(
            os.path.join(metadata_path, "validation_indices")
        )
        test_indices = torch.load(os.path.join(metadata_path, "test_indices"))
    else:
        all_indices = torch.randperm(len(holdout_set))
        val_indices = all_indices[
            : datasets_metadata[dataset_name]["validation_size"]
        ]
        torch.save(
            val_indices, os.path.join(output_path, "validation_indices")
        )
        test_indices = all_indices[
            datasets_metadata[dataset_name]["validation_size"] :
        ]
        torch.save(test_indices, os.path.join(output_path, "test_indices"))

    val_set = Subset(holdout_set, val_indices)
    test_set = Subset(holdout_set, test_indices)

    dataset_handler_functions = {
        "mixed": (
            handle_mixed_dataset,
            {
                "adversarial_cls": adversarial_cls,
                "adversarial_dir": adversarial_dir,
            },
        ),
        "subclass": (handle_subclass_dataset, {"num_groups": num_groups}),
        "mislabeled": (
            handle_mislabeled_dataset,
            {"mislabeling_probability": mislabeling_probability},
        ),
        "shortcut": (
            handle_shortcut_dataset,
            {
                "shortcut_cls": shortcut_cls,
                "shortcut_probability": shortcut_probability,
            },
        ),
    }
    if dataset_type != "vanilla":
        fn, kwargs = dataset_handler_functions[dataset_type]
        train_set, val_set, test_set, extra_info = fn(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            metadata_path=metadata_path,
            dataset_name=dataset_name,
            output_path=output_path,
            num_classes=num_classes,
            regular_transforms=regular_transforms,
            seed=seed,
            **kwargs,
        )
    else:
        extra_info = {}

    return_dict = {
        "validation_indices": val_indices,
        "test_indices": test_indices,
        **extra_info,
    }
    return train_set, val_set, test_set, return_dict


def load_pl_module(
    module_name: str,
    pretrained: bool,
    epochs: int,
    num_outputs: int,
    device: str,
):
    module_kwargs = {
        "MnistModel": {"epochs": epochs},
        "TinyImagenetModel": {"pretrained": pretrained},
    }
    module = load_module_with_name(
        module_name=module_name,
        num_outputs=num_outputs,
        device=device,
        **module_kwargs[module_name],
    )
    return module


def train_model(
    dataset_name: str,
    dataset_cache_dir: str,
    augmentation: str,
    dataset_type: str,
    metadata_root: str,
    output_path: str,
    seed: int,
    adversarial_dir: str,
    device: str,
    module_name: str,
    pretrained: bool,
    epochs: int,
    lr: float,
    batch_size: int,
    validate_each: int,
    save_each: int,
    weight_decay: float,
    model_path: str,
    base_epoch: int,
):
    torch.set_float32_matmul_precision("medium")
    torch.manual_seed(seed)
    os.makedirs(output_path, exist_ok=True)
    save_id_base = f"{dataset_type}_{lr}_{weight_decay}{f'_{augmentation}' if augmentation is not None else ''}{'_pre' if pretrained else ''}"

    if save_each is None:
        save_each = validate_each

    if output_path is None:
        output_path = os.path.join(metadata_root, dataset_name)
    os.makedirs(os.path.join(metadata_root, dataset_name), exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # Load train and validation datasets
    train_set, val_set, _, _ = load_datasets(
        dataset_name=dataset_name,
        dataset_cache_dir=dataset_cache_dir,
        augmentation=augmentation,
        dataset_type=dataset_type,
        metadata_path=os.path.join(metadata_root, dataset_name),
        output_path=output_path,
        seed=seed,
        adversarial_dir=adversarial_dir,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    num_outputs = datasets_metadata[dataset_name][
        "num_groups" if dataset_type == "subclass" else "num_classes"
    ]
    pl_module = load_pl_module(
        module_name=module_name,
        epochs=epochs,
        pretrained=pretrained,
        num_outputs=num_outputs,
        device=device,
    )
    if model_path is not None:
        pl_module.load_state_dict(torch.load(model_path, map_location=device))
    pl_module.to(device)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        filename=save_id_base + "_{epoch}",
        every_n_epochs=save_each,
        save_top_k=-1,
    )
    logger = TensorBoardLogger(output_path, name=save_id_base)
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=base_epoch + epochs,
        check_val_every_n_epoch=validate_each,
        default_root_dir=output_path,
        progress_bar_refresh_rate=0,
        logger=logger,
    )
    trainer.fit(pl_module, train_loader, val_loader)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Define argument for method with choices
    parser.add_argument(
        "--dataset_name",
        required=True,
        default="mnist",
        type=str,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        required=False,
        default=None,
        type=str,
        help="Directory to cache HF datasets",
    )
    parser.add_argument(
        "--augmentation",
        required=False,
        type=str,
        default=None,
        help="Augmentation tag composed of keywords seperated with _",
    )
    parser.add_argument(
        "--dataset_type",
        required=False,
        default="vanilla",
        choices=["vanilla", "mislabeled", "shortcut", "mixed", "subclass"],
    )
    parser.add_argument(
        "--metadata_root",
        required=True,
        type=str,
        help="Path to metadata directory",
    )
    parser.add_argument(
        "--output_path",
        required=False,
        type=str,
        default=None,
        help="Directory to save outputs",
    )
    parser.add_argument("--seed", required=False, type=int, default=42)
    parser.add_argument(
        "--adversarial_dir", required=False, type=str, default=None
    )
    parser.add_argument(
        "--device",
        required=False,
        type=str,
        help="Device to run the model on",
        choices=["cpu", "cuda"],
        default=None,
    )
    parser.add_argument(
        "--module_name", required=True, type=str, default="MnistModel"
    )
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", required=True, type=int, default=100)
    parser.add_argument("--lr", required=True, type=float, default=0.1)
    parser.add_argument("--batch_size", required=True, type=int, default=64)
    parser.add_argument("--validate_each", required=True, type=int, default=10)
    parser.add_argument("--save_each", required=False, type=int, default=None)
    parser.add_argument(
        "--weight_decay", required=False, type=float, default=0.0
    )
    parser.add_argument(
        "--model_path",
        required=False,
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument("--base_epoch", required=False, type=int, default=0)

    args = parser.parse_args()

    # Call the function with parsed arguments
    train_model(
        dataset_name=args.dataset_name,
        dataset_cache_dir=args.dataset_cache_dir,
        augmentation=args.augmentation,
        dataset_type=args.dataset_type,
        metadata_root=args.metadata_root,
        output_path=args.output_path,
        seed=args.seed,
        adversarial_dir=args.adversarial_dir,
        device=args.device,
        module_name=args.module_name,
        pretrained=args.pretrained,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        validate_each=args.validate_each,
        save_each=args.save_each,
        weight_decay=args.weight_decay,
        model_path=args.model_path,
        base_epoch=args.base_epoch,
    )
