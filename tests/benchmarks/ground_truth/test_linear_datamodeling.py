import os
from copy import deepcopy

import pytest
import torch
import yaml

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.ground_truth import LinearDatamodeling
from quanda.benchmarks.resources import config_map
from quanda.metrics.ground_truth.linear_datamodeling import (
    LinearDatamodelingMetric,
)


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.parametrize(
    "config_name,subset_acc_threshold",
    [
        ("mnist_linear_datamodeling", 0.9),
        ("cifar_linear_datamodeling", 0.7),
    ],
)
def test_lds_sanity_check_subset_accuracy(
    config_name, subset_acc_threshold, tmp_path
):
    """Verify that all subset checkpoints achieve > threshold accuracy
    on the eval dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    bench = LinearDatamodeling.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    sanity_results = bench.sanity_check(batch_size=batch_size)

    subset_accs = [
        sanity_results[acc]
        for acc in sanity_results
        if acc.startswith("subset_acc_")
    ]
    assert len(subset_accs) == bench.m, (
        f"Expected {bench.m} subset accuracies, got {len(subset_accs)}."
    )
    for i, acc in enumerate(subset_accs):
        assert acc > subset_acc_threshold, (
            f"Subset checkpoint {i} accuracy {acc:.4f} "
            f"is below the {subset_acc_threshold} threshold."
        )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_linear_datamodeling",
        "cifar_linear_datamodeling",
    ],
)
def test_lds_subset_checkpoints_are_different(config_name, tmp_path):
    """Verify that all subset checkpoint state dicts are pairwise
    different, ensuring each subset model was trained independently."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bench = LinearDatamodeling.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    state_dicts = []
    for ckpt_path in bench.subset_ckpt_filenames[:5]:
        subset_model = deepcopy(bench.model)
        bench.checkpoints_load_func(subset_model, ckpt_path)
        state_dicts.append(subset_model.state_dict())

    for i in range(len(state_dicts)):
        for j in range(i + 1, len(state_dicts)):
            all_equal = all(
                torch.equal(state_dicts[i][k], state_dicts[j][k])
                for k in state_dicts[i]
            )
            assert not all_equal, (
                f"Subset checkpoints {i} and {j} have identical "
                f"state dicts — they should be different."
            )


@pytest.mark.utils
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_linear_datamodeling",
        "cifar_linear_datamodeling",
    ],
)
def test_lds_metadata(
    config_name,
    tmp_path,
    request,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bench = LinearDatamodeling.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    bench_yaml = config_map[config_name]
    with open(bench_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    metadata_dir = BenchConfigParser.get_metadata_dir(
        cfg=cfg, bench_save_dir=str(tmp_path)
    )

    subset_meta = f"{metadata_dir}/{cfg['subset_ids']}"
    with open(subset_meta, "r") as f:
        subset_ids = yaml.safe_load(f)

    assert subset_ids == bench.subset_ids


@pytest.mark.benchmarks
def test_lds_metric_uses_subset_ckpt_filenames(
    load_mnist_model,
    load_mnist_dataset,
    load_mnist_test_samples_1,
    load_mnist_test_labels_1,
    load_subset_indices_lds,
    load_pretrained_models_lds,
):
    """Verify LDS metric loads subset models lazily from
    subset_ckpt_filenames rather than holding them all in memory."""
    test_data = load_mnist_test_samples_1
    test_targets = torch.tensor(load_mnist_test_labels_1)

    with open(
        f"tests/assets/lds_checkpoints/{load_subset_indices_lds}", "r"
    ) as f:
        subset_ids = yaml.safe_load(f)

    metric = LinearDatamodelingMetric(
        model=load_mnist_model,
        train_dataset=load_mnist_dataset,
        alpha=0.5,
        model_id="mnist_lds",
        m=len(load_pretrained_models_lds),
        seed=3,
        correlation_fn="spearman",
        cache_dir="tests/assets/lds_checkpoints/",
        batch_size=1,
        subset_ids=subset_ids,
        subset_ckpt_filenames=load_pretrained_models_lds,
    )

    assert metric.subset_ckpt_filenames == load_pretrained_models_lds

    explanations = torch.randn(
        test_data.shape[0], len(load_mnist_dataset)
    )
    metric.update(
        test_data=test_data,
        explanations=explanations,
        test_targets=test_targets,
    )
    score = metric.compute()["score"]
    assert isinstance(score, float)


@pytest.mark.benchmarks
def test_lds_metric_missing_checkpoints_raises(
    load_mnist_model,
    load_mnist_dataset,
    load_subset_indices_lds,
):
    with open(
        f"tests/assets/lds_checkpoints/{load_subset_indices_lds}", "r"
    ) as f:
        subset_ids = yaml.safe_load(f)

    with pytest.raises(FileNotFoundError):
        LinearDatamodelingMetric(
            model=load_mnist_model,
            train_dataset=load_mnist_dataset,
            alpha=0.5,
            m=1,
            seed=3,
            correlation_fn="spearman",
            subset_ids=subset_ids,
            subset_ckpt_filenames=["/nonexistent/path/model.pt"],
        )
