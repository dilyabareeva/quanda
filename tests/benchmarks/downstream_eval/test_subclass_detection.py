import os

import pytest
import torch

from quanda.benchmarks.downstream_eval import (
    SubclassDetection,
)


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_subclass_detection",
    ],
)
def test_subclass_class_to_group(config_name, tmp_path):
    """Verify subclass-transformed eval samples stay unchanged
    after re-applying the sample transform."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench = SubclassDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    datasets = [bench.train_dataset, bench.eval_dataset]
    class_to_group = {}

    mismatches = []
    for ds in datasets:
        for idx in range(len(ds)):
            raw_label = ds.get_original_label(idx)
            _, label = ds[idx]
            if raw_label in class_to_group:
                if class_to_group[raw_label] != label:
                    mismatches.append(
                        (idx, raw_label, class_to_group[raw_label], label)
                    )
            else:
                class_to_group[raw_label] = label

    assert not mismatches, (
        f"Found {len(mismatches)} samples where raw label mapped "
        f"to different transformed labels: {mismatches[:10]}"
    )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name,expected_thresholds",
    [
        (
            "mnist_subclass_detection",
            {
                "train_acc": 0.9,
                "val_acc": 0.9,
                "eval_post_filter_percentage": 0.5,
            },
        ),
        (
            "cifar_subclass_detection",
            {
                "train_acc": 0.9,
                "val_acc": 0.9,
                "eval_post_filter_percentage": 0.5,
            },
        ),
    ],
)
def test_subclass_sanity_check_values(
    config_name, expected_thresholds, tmp_path
):
    """Verify filter_by_non_subclass and filter_by_shortcut_pred in benchmark cfg work as expected on eval_dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    bench = SubclassDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    sanity_check_results = bench.sanity_check(batch_size=batch_size)

    for key, threshold in expected_thresholds.items():
        assert sanity_check_results[key] > threshold, (
            f"Expected {key} > {threshold}, got {sanity_check_results[key]}."
        )
