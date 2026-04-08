import os
from copy import deepcopy

import pytest
import torch
import yaml

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.ground_truth import LinearDatamodeling
from quanda.benchmarks.resources import config_map


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_linear_datamodeling",
    ],
)
def test_lds_sanity_check_subset_accuracy(config_name, tmp_path):
    """Verify that all subset checkpoints achieve > 90% accuracy
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

    subset_accs = sanity_results["subset_accs"]
    assert len(subset_accs) == bench.m, (
        f"Expected {bench.m} subset accuracies, "
        f"got {len(subset_accs)}."
    )
    for i, acc in enumerate(subset_accs):
        assert acc > 0.9, (
            f"Subset checkpoint {i} accuracy {acc:.4f} "
            f"is below the 0.9 threshold."
        )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_linear_datamodeling",
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
    for ckpt_path in bench.subset_ckpt_filenames:
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
    ],
)
def test_lds_metadata(
    config_name,
    tmp_path,
    request,
):

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
        cfg=cfg, bench_save_dir=cfg.get("bench_save_dir", "./tmp")
    )

    subset_meta = f"{metadata_dir}/{cfg['subset_ids']}"
    with open(subset_meta, "r") as f:
        subset_ids = yaml.safe_load(f)
        
    bench_subset_ids = bench.subset_ids

    assert subset_ids == bench_subset_ids
