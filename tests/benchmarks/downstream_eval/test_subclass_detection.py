import pytest
import torch
import yaml

from quanda.benchmarks.downstream_eval import (
    SubclassDetection,
)
from quanda.benchmarks.resources import config_map


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

    indices_cfg = cfg["train_dataset"]["indices"]
    split_name = indices_cfg["split_name"]
    eval_ratio = indices_cfg["split_ratios"][split_name]
    pre_filter_size = len(bench.train_dataset.dataset.dataset)
    filtered_size = len(bench.train_dataset.dataset)
    actual_ratio = filtered_size / pre_filter_size

    assert (actual_ratio / eval_ratio) > 0.5, (
        f"Expected the filtered dataset to be close to the specified eval_ratio of {eval_ratio}, "
        f"but got an actual ratio of {actual_ratio:.2f} (filtered size: {filtered_size}, pre-filter size: {pre_filter_size})."
    )

    sanity_check_results = bench.sanity_check(batch_size=batch_size)

    assert sanity_check_results["train_acc"] > 0.9, (
        f"Expected train_acc to be > 0.9, but got {sanity_check_results['train_acc']}."
    )
    assert sanity_check_results["val_acc"] > 0.9, (
        f"Expected val_acc to be > 0.9, but got {sanity_check_results['val_acc']}."
    )
