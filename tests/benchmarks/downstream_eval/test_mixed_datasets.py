import os

import pytest
import torch
import yaml

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.heuristics.mixed_datasets import MixedDatasets
from quanda.benchmarks.resources import config_map


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_mixed_datasets",
    ],
)
def test_train_dataset_indexing_is_correct(config_name, tmp_path):
    """Verify subclass-transformed eval samples stay unchanged
    after re-applying the sample transform."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bench_yaml = config_map[config_name]
    with open(bench_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    metadata_dir = BenchConfigParser.get_metadata_dir(
        cfg=cfg,
        bench_save_dir=cfg.get("bench_save_dir", "./tmp"),
    )
    bench = MixedDatasets.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    full_adv_dataset = BenchConfigParser.parse_dataset_cfg(
        ds_config=cfg["adv_dataset"],
        metadata_dir=metadata_dir,
    )
    split_datasets = BenchConfigParser.split_dataset(
        dataset=full_adv_dataset,
        ds_config=cfg["adv_dataset"],
        metadata_dir=metadata_dir,
    )
    adv_base_dataset = split_datasets["train"]

    train_ds = bench.train_dataset

    adv_indices = [
        i for i, v in enumerate(bench.adversarial_indices) if v == 1
    ]
    mismatches = []
    for idx in range(len(adv_indices)):
        raw_item, _ = adv_base_dataset[idx]
        train_item, _ = train_ds[adv_indices[idx]]
        if not torch.allclose(raw_item, train_item):
            mismatches.append(idx)

    assert not mismatches, (
        f"Found {len(mismatches)} samples where the adversarial sample "
        f"in the train dataset does not match the corresponding sample in "
        f"the original adv_base_dataset. "
    )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name,expected_thresholds",
    [
        (
            "mnist_mixed_datasets",
            {
                "train_acc": 0.85,
                "val_acc": 0.85,
                "train_adversarial_memorization": 0.8,
                "eval_adversarial_memorization": 0.8,
            },
        ),
        (
            "cifar_mixed_datasets",
            {
                "train_acc": 0.85,
                "val_acc": 0.84,
                "train_adversarial_memorization": 0.75,
                "eval_adversarial_memorization": 0.74,
            },
        ),
    ],
)
def test_mixed_datasets_sanity_check_values(
    config_name, expected_thresholds, tmp_path
):
    """Verify filter_by_non_subclass and filter_by_shortcut_pred in benchmark cfg work as expected on eval_dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    bench_yaml = config_map[config_name]
    with open(bench_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    bench = MixedDatasets.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    eval_ratio = cfg["adv_dataset"]["indices"]["split_ratios"]["val"]
    pre_filter_size = len(bench.eval_dataset.dataset.dataset)
    filtered_size = len(bench.eval_dataset)
    actual_ratio = filtered_size / pre_filter_size

    assert (actual_ratio / eval_ratio) > 0.5, (
        f"Expected the filtered dataset to be close to the specified eval_ratio of {eval_ratio}, "
        f"but got an actual ratio of {actual_ratio:.2f} (filtered size: {filtered_size}, pre-filter size: {pre_filter_size})."
    )

    sanity_check_results = bench.sanity_check(batch_size=batch_size)

    for key, threshold in expected_thresholds.items():
        assert sanity_check_results[key] > threshold, (
            f"Expected {key} > {threshold}, "
            f"got {sanity_check_results[key]}."
        )
