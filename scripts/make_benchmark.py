import logging
import os
import sys

import torch.utils

sys.path.append(os.getcwd())

from argparse import ArgumentParser

from quanda.metrics.ground_truth import LinearDatamodelingMetric

import torch
import lightning as L
from scripts.train_model import (
    load_datasets,
    datasets_metadata,
    load_pl_module,
)

from quanda.benchmarks.resources.modules import bench_load_state_dict

logger = logging.getLogger(__name__)


def get_dataset_type(benchmark_name):
    if benchmark_name == "mislabeling_detection":
        return "mislabeled"
    if benchmark_name == "subclass_detection":
        return "subclass"
    if benchmark_name == "shortcut_detection":
        return "shortcut"
    if benchmark_name == "mixed_datasets":
        return "mixed"
    return "vanilla"


checkpoints_urls = {
    "vanilla": "https://datacloud.hhi.fraunhofer.de/s/NKLynmg3dSrnS4A/download",
    "mislabeled": "https://datacloud.hhi.fraunhofer.de/s/6tXPimRgSNQ44dt/download",  # old
    "shortcut": "https://datacloud.hhi.fraunhofer.de/s/ky993WCebZGBYBr/download",  # OLD
    "mixed": "https://datacloud.hhi.fraunhofer.de/s/QAwjQBLNY8Mt3Zj/download",
    "subclass": "https://datacloud.hhi.fraunhofer.de/s/ckysrWWQq3J4WET/download",
}


def make_benchmark(
    benchmark_name: str,
    dataset_name: str,
    dataset_cache_dir: str,
    metadata_root: str,
    output_path: str,
    seed: int,
    adversarial_dir: str,
    device: str,
    module_name: str,
    checkpoints_dir: str,
):
    torch.set_float32_matmul_precision("medium")
    # EVAL_SET_SIZE = 128
    torch.manual_seed(seed)
    dataset_type = get_dataset_type(benchmark_name)
    checkpoints_dir = os.path.join(checkpoints_dir, dataset_type)
    if output_path is None:
        output_path = os.path.join(metadata_root, dataset_name)
    os.makedirs(metadata_root, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    train_set, val_set, test_set, ds_dict = load_datasets(
        dataset_name=dataset_name,
        dataset_cache_dir=dataset_cache_dir,
        augmentation=None,
        dataset_type=dataset_type,
        metadata_path=os.path.join(metadata_root, dataset_name),
        output_path=output_path,
        seed=seed,
        adversarial_dir=adversarial_dir,
    )

    num_outputs = datasets_metadata[dataset_name][
        "num_groups" if dataset_type == "subclass" else "num_classes"
    ]

    ckpt_names = []
    ckpt_binary = []
    file_list = [f for f in os.listdir(checkpoints_dir) if f.endswith(".ckpt")]
    file_list = sorted(
        file_list, key=lambda x: int(x.split("=")[1].split(".")[0])
    )
    for file in file_list:
        ckpt_names.append(file)
        # epoch = int(
        #     file.replace("epoch=", "").replace(".ckpt", "").split("_")[-1]
        # )
        model_state_dict = torch.load(
            os.path.join(checkpoints_dir, file), map_location=device
        )
        # EDIT HERE TO CHANGE THE CONTENTS OF CHECKPOINT BINARY
        ckpt_binary.append(model_state_dict)

    ckpts_url = checkpoints_urls[dataset_type]

    bench_state = {
        "test_split_name": datasets_metadata[dataset_name]["test_split_name"],
        "dataset_str": datasets_metadata[dataset_name]["hf_tag"],
        "checkpoints": ckpt_names,
        "checkpoints_url": ckpts_url,
        "use_predictions": True,
        "n_classes": num_outputs,
        "eval_test_indices": ds_dict["test_indices"],
        "dataset_transform": f"{dataset_name}_transform",
        "checkpoints_dir_name": f"{dataset_name}_{module_name}_{benchmark_name}_benchmark_checkpoints",
        "pl_module": module_name,
    }

    if benchmark_name == "subclass_detection":
        bench_state["class_to_group"] = ds_dict["class_to_group"]
    elif benchmark_name == "mislabeling_detection":
        bench_state["mislabeling_labels"] = ds_dict["mislabeling_labels"]
    elif benchmark_name == "shortcut_detection":
        bench_state["shortcut_cls"] = ds_dict["shortcut_cls"]
        bench_state["shortcut_indices"] = ds_dict["shortcut_indices"]
        bench_state["sample_fn"] = f"{dataset_name}_shortcut_transform"
    elif benchmark_name == "mixed_datasets":
        bench_state["adversarial_label"] = ds_dict["adversarial_cls"]
        bench_state["adversarial_transform"] = (
            f"{dataset_name}_adversarial_transform"
        )
        bench_state["adv_train_indices"] = ds_dict["adversarial_train_indices"]
        bench_state["adv_test_indices"] = ds_dict["adversarial_test_indices"]
        bench_state["adversarial_dir_url"] = ds_dict["adversarial_dir_url"]
        bench_state["eval_test_indices"] = list(
            range(len(ds_dict["adversarial_test_indices"]))
        )  # see base benchmark's _parse_bench_state
    elif benchmark_name == "linear_datamodeling":
        bench_state["m"] = 100
        bench_state["alpha"] = 0.5
        bench_state["trainer_fit_kwargs"] = {}
        bench_state["correlation_fn"] = "spearman"
        bench_state["model_id"] = f"{dataset_name}_{module_name}_0"
        bench_state["seed"] = seed
        module = load_pl_module(
            module_name=bench_state["pl_module"],
            num_outputs=num_outputs,
            epochs=100,
            lr=0.001,
            pretrained=True,
            device=device,
        )
        metric = LinearDatamodelingMetric(
            model=module,
            train_dataset=train_set,
            trainer=L.Trainer(max_epochs=20),
            alpha=bench_state["alpha"],
            m=bench_state["m"],
            correlation_fn=bench_state["correlation_fn"],
            trainer_fit_kwargs=bench_state["trainer_fit_kwargs"],
            seed=bench_state["seed"],
            checkpoints=ckpt_binary,
            checkpoints_load_func=bench_load_state_dict,
            cache_dir=output_path,
        )
        bench_state["subset_ids"] = metric.subsets
        assert len(bench_state["subset_ids"]) == len(bench_state["m"])

        pretrained_ckpts = []

        for i in range(bench_state["m"]):
            model_ckpt_path = os.path.join(
                output_path, f"{bench_state['model_id']}_model_{i}.ckpt"
            )
            ckpt = torch.load(model_ckpt_path, map_location=device)
            pretrained_ckpts.append(ckpt)
        bench_state["pretrained_model_checkpoints"] = pretrained_ckpts

    torch.save(
        bench_state,
        os.path.join(output_path, f"{dataset_name}_{benchmark_name}.pth"),
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    # Define argument for method with choices
    parser.add_argument(
        "--benchmark_name",
        required=True,
        default="class_detection",
        type=str,
        help="Name of the dataset",
    )
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
        "--module_name",
        required=True,
        default="MnistModel",
        type=str,
        help="Name of the Lightning module",
    )
    parser.add_argument(
        "--checkpoints_dir",
        required=False,
        default=None,
        type=str,
        help="Directory used to load checkpoint binaries",
    )
    args = parser.parse_args()

    # Call the function with parsed arguments
    make_benchmark(
        args.benchmark_name,
        args.dataset_name,
        args.dataset_cache_dir,
        args.metadata_root,
        args.output_path,
        args.seed,
        args.adversarial_dir,
        args.device,
        args.module_name,
        args.checkpoints_dir,
    )
