import os

import pytest
import torch
import yaml

from quanda.benchmarks.downstream_eval import (
    SubclassDetection,
)
from quanda.benchmarks.resources import config_map


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
# @pytest.mark.production_bench
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
# @pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_subclass_detection",
    ],
)
def test_subclass_sanity_check_values(config_name, tmp_path):
    """Verify filter_by_non_subclass and filter_by_shortcut_pred in benchmark cfg work as expected on eval_dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    bench_yaml = config_map[config_name]
    with open(bench_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    bench = SubclassDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    sanity_check_results = bench.sanity_check(batch_size=batch_size)

    assert sanity_check_results["train_acc"] > 0.9, (
        f"Expected train_acc to be > 0.9, but got {sanity_check_results['train_acc']}."
    )
    assert sanity_check_results["val_acc"] > 0.9, (
        f"Expected val_acc to be > 0.9, but got {sanity_check_results['val_acc']}."
    )

    assert (
        sanity_check_results["eval_post_filter_percentage"] > 0.5
    ), (  # TODO: retrain until this improves (
        f"Expected eval_post_filter_percentage to be > 0.5, but got {sanity_check_results['eval_post_filter_percentage']}."
    )
