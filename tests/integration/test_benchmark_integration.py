# START1
import os

import pytest
import torch
import torchvision
import yaml

from quanda.benchmarks.downstream_eval import (
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


# @pytest.mark.slow
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

    # Subset datasets for faster testing
    n_train, n_eval = 50, 20
    sc_idx = list(benchmark.train_dataset.transform_indices[:10])
    other_idx = [
        i for i in range(len(benchmark.train_dataset)) if i not in set(sc_idx)
    ][: n_train - len(sc_idx)]
    benchmark.train_dataset.apply_filter(sorted(sc_idx + other_idx))

    eval_sc_idx = list(benchmark.eval_dataset.transform_indices[:5])
    eval_other = [
        i
        for i in range(len(benchmark.eval_dataset))
        if i not in set(eval_sc_idx)
    ][: n_eval - len(eval_sc_idx)]
    benchmark.eval_dataset.apply_filter(sorted(eval_sc_idx + eval_other))

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

    # Override for faster testing
    captum_influence_args["projection_dim"] = 2
    captum_influence_args["arnoldi_dim"] = 5
    captum_influence_args["arnoldi_tol"] = 1e-10
    captum_influence_args["hessian_dataset"] = torch.utils.data.Subset(
        benchmark.train_dataset,
        torch.randint(0, len(benchmark.train_dataset), (50,)),
    )

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

    # Override for faster testing
    trak_args["proj_dim"] = 32

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

    # Override for faster testing
    representer_points_args["epoch"] = 1

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

    # START14_2
    with open(
        "tests/assets/mnist_local_bench/20fba38-default_SubclassDetection.yaml",
        "r",
    ) as f:
        subclass_config = yaml.safe_load(f)
    # END14_2

    # Override for faster testing
    subclass_config["model"]["trainer"]["max_epochs"] = 2

    # START14_3
    subclass_config["bench_save_dir"] = os.path.join(
        cache_dir, "subclass_detection_bench"
    )
    benchmark = SubclassDetection.train(
        subclass_config,
        device=device,
    )
    # END14_3

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
