import pytest
from torch.utils.data import random_split

# START1
import os
import sys
import torch
import torchvision
from quanda.benchmarks.downstream_eval import ShortcutDetection, MislabelingDetection, SubclassDetection
from quanda.explainers.wrappers import (
    TRAK,
    CaptumArnoldi,
    CaptumSimilarity,
    CaptumTracInCPFast,
    RepresenterPoints,
)
# END1

# START14_1
from quanda.benchmarks.resources import pl_modules
import lightning as L
# END14_1


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id",
    [
        (
            "benchmark_tutorial",
        )
    ],
)
def test_benchmarks(
    test_id,
):
    # START2
    torch.set_float32_matmul_precision("medium")
    to_img = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(mean=0.0, std=2.0),
            torchvision.transforms.Normalize(mean=-0.5, std=1.0),
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
        ]
    )
    # END2

    # START3
    cache_dir = str(os.path.join(
        os.getcwd(), "quanda_benchmark_tutorial_cache"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark = ShortcutDetection.download(
        name="mnist_shortcut_detection",
        cache_dir=cache_dir,
        device=device,
    )
    # END3

    benchmark.base_dataset = torch.utils.data.Subset(
        benchmark.base_dataset, list(range(16))
    )
    benchmark.shortcut_dataset = torch.utils.data.Subset(
        benchmark.shortcut_dataset, list(range(16))
    )
    benchmark.eval_dataset = torch.utils.data.Subset(
        benchmark.eval_dataset, list(range(16))
    )
    benchmark.shortcut_indices = [
        i for i in benchmark.shortcut_indices if i < 16
    ]

    # START4
    shortcut_img = benchmark.shortcut_dataset[benchmark.shortcut_indices[0]][0]
    tensor_img = torch.concat(
        [shortcut_img, shortcut_img, shortcut_img], dim=0)
    img = to_img(tensor_img)
    # END4

    # START5
    captum_similarity_args = {
        "model_id": "mnist_shortcut_detection_tutorial",
        "layers": "model.fc_2",
        "cache_dir": os.path.join(cache_dir, "captum_similarity"),
    }
    # END5

    # START6
    hessian_num_samples = 500  # number of samples to use for hessian estimation
    hessian_ds = torch.utils.data.Subset(
        benchmark.shortcut_dataset, torch.randint(
            0, len(benchmark.shortcut_dataset), (hessian_num_samples,))
    )

    captum_influence_args = {
        "layers": ["model.fc_3"],
        "batch_size": 8,
        "hessian_dataset": hessian_ds,
        "projection_dim": 5,
    }
    # END6

    # START7
    captum_tracin_args = {
        "final_fc_layer": "model.fc_3",
        "loss_fn": torch.nn.CrossEntropyLoss(reduction="mean"),
        "batch_size": 8,
    }
    # END7

    # START8
    trak_args = {
        "model_id": "mnist_shortcut_detection",
        "cache_dir": os.path.join(cache_dir, "trak"),
        "batch_size": 8,
        "proj_dim": 2048,
    }
    # END8

    # START9
    representer_points_args = {
        "model_id": "mnist_shortcut_detection",
        "cache_dir": os.path.join(cache_dir, "representer_points"),
        "batch_size": 8,
        "features_layer": "model.relu_4",
        "classifier_layer": "model.fc_3",
    }
    # END9

    # START10
    attributors = {
        "captum_similarity": (CaptumSimilarity, captum_similarity_args),
        "captum_arnoldi": (CaptumArnoldi, captum_influence_args),
        "captum_tracin": (CaptumTracInCPFast, captum_tracin_args),
        "trak": (TRAK, trak_args),
        "representer": (RepresenterPoints, representer_points_args),
    }
    results = dict()
    for name, (cls, kwargs) in attributors.items():
        print(name)
        results[name] = benchmark.evaluate(
            explainer_cls=cls, expl_kwargs=kwargs, batch_size=8)["score"]
    # END10

    # START11
    temp_benchmark = MislabelingDetection.download(
        name="mnist_mislabeling_detection",
        cache_dir=cache_dir,
        device=device,
    )
    # END11

    # START12
    model = temp_benchmark.model
    base_dataset = temp_benchmark.base_dataset
    mislabeling_labels = temp_benchmark.mislabeling_labels
    dataset_transform = None
    # END12

    # START13
    benchmark = MislabelingDetection.assemble(
        model=model,
        base_dataset=base_dataset,
        n_classes=10,
        mislabeling_labels=mislabeling_labels,
        dataset_transform=dataset_transform,
        device=device,
    )
    representer_points_args = {
        "model_id": "mnist_mislabeling_detection",
        "cache_dir": os.path.join(cache_dir, "representer_points"),
        "batch_size": 8,
        "features_layer": "model.relu_4",
        "classifier_layer": "model.fc_3",
    }
    results = benchmark.evaluate(
        explainer_cls=RepresenterPoints,
        expl_kwargs=representer_points_args,
    )
    # END13

    # START14_2
    num_groups = 2
    model = pl_modules["MnistModel"](num_labels=num_groups, device=device)
    trainer = L.Trainer(max_epochs=10)
    dataset_transform = None

    # Collect base and evaluation datasets from a precomputed benchmark for simplicity, instead of creating the dataset objects from scratch
    base_dataset = temp_benchmark.base_dataset
    eval_dataset = temp_benchmark.eval_dataset

    benchmark = SubclassDetection.generate(
        model=model,
        cache_dir="./cache",
        trainer=trainer,
        base_dataset=base_dataset,
        eval_dataset=eval_dataset,
        dataset_transform=dataset_transform,
        n_classes=10,
        n_groups=num_groups,
        class_to_group="random",
    )
    # END14_2

    # START15
    results = benchmark.evaluate(
        explainer_cls=CaptumSimilarity,
        expl_kwargs={
            "model_id": "mnist_subclass_detection_tutorial",
            "layers": "model.fc_2",
            "cache_dir": os.path.join(cache_dir, "captum_similarity"),
        },
    )
    # END15
