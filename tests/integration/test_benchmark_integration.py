# START1
import os

import pytest
import torch
import torchvision
import yaml

from quanda.benchmarks.downstream_eval import (
    MislabelingDetection,
    ShortcutDetection,
    SubclassDetection,
)

# END1
# START14_1
from quanda.explainers.wrappers import (
    TRAK,
    CaptumArnoldi,
    CaptumSimilarity,
    CaptumTracInCPFast,
    RepresenterPoints,
)

# END14_1


@pytest.mark.skip(reason="Slow benchmark integration test")
@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize(
    "test_id",
    [("benchmark_tutorial",)],
)
def test_benchmark_integration(
    test_id,
    tmp_path,
):
    torch.manual_seed(42)

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

    cache_dir = str(os.path.join(tmp_path, "quanda_benchmark_tutorial_cache"))

    # START3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    benchmark = ShortcutDetection.load_pretrained(
        bench_id="mnist_shortcut_detection",
        cache_dir=cache_dir,
        device=device,
    )
    # END3

    # START4
    shortcut_img = benchmark.train_dataset[
        benchmark.train_dataset.transform_indices[0]
    ][0]
    tensor_img = shortcut_img.repeat(3, 1, 1)
    img = to_img(tensor_img)
    img.show(title="Shortcut Image")
    # END4

    """
    benchmark.shortcut_indices = benchmark.train_dataset.transform_indices[:4]

    benchmark.train_dataset = torch.utils.data.Subset(
        benchmark.train_dataset, list(range(4))
    )
    benchmark.val_dataset = torch.utils.data.Subset(
        benchmark.val_dataset, list(range(4))
    )
    benchmark.eval_dataset = torch.utils.data.Subset(
        benchmark.eval_dataset, list(range(4))
    )
    """

    # START5
    captum_similarity_args = {
        "model_id": "mnist_shortcut_detection_tutorial",
        "layers": "fc_2",
        "cache_dir": os.path.join(cache_dir, "captum_similarity"),
    }
    # END5

    # START6
    hessian_num_samples = (
        500  # number of samples to use for hessian estimation
    )
    hessian_ds = torch.utils.data.Subset(
        benchmark.train_dataset,
        torch.randint(0, len(benchmark.train_dataset), (hessian_num_samples,)),
    )

    captum_influence_args = {
        "layers": ["fc_3"],
        "batch_size": 8,
        "hessian_dataset": hessian_ds,
        "projection_dim": 5,
    }
    # END6

    # START7
    captum_tracin_args = {
        "final_fc_layer": "fc_3",
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
        "epoch": 100,
        "features_layer": "relu_4",
        "classifier_layer": "fc_3",
    }
    # END9

    # TODO: Add TracIn after the checkpoint loading issue is resolved
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
        results[name] = benchmark.evaluate(
            explainer_cls=cls, expl_kwargs=kwargs, batch_size=8
        )["score"]
    # END10

    # START11
    benchmark = MislabelingDetection.load_pretrained(
        bench_id="mnist_mislabeling_detection_unit",
        cache_dir=cache_dir,
    )
    # END11
    print(type(benchmark))

    # START13
    with open(
        "tests/assets/mnist_test_suite_2/7ed30b3-default_MislabelingDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)

    benchmark = MislabelingDetection.from_config(
        config,
    )
    representer_points_args = {
        "model_id": "mnist_mislabeling_detection",
        "cache_dir": os.path.join(cache_dir, "representer_points"),
        "batch_size": 8,
        "epoch": 100,
        "features_layer": "relu_4",
        "classifier_layer": "fc_3",
    }
    results = benchmark.evaluate(
        explainer_cls=RepresenterPoints,
        expl_kwargs=representer_points_args,
    )
    # END13

    # START14_2
    with open(
        "tests/assets/mnist_test_suite_2/7ed30b3-default_SubclassDetection.yaml",
        "r",
    ) as f:
        subclass_config = yaml.safe_load(f)

    benchmark = SubclassDetection.train(
        subclass_config,
        device=device,
    )
    # END14_2

    # START15
    results = benchmark.evaluate(
        explainer_cls=CaptumSimilarity,
        expl_kwargs={
            "model_id": "mnist_subclass_detection_tutorial",
            "layers": "fc_2",
            "cache_dir": os.path.join(cache_dir, "captum_similarity"),
        },
    )
    # END15
