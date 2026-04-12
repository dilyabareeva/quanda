import math
import os

import pytest
import torch
import yaml

from quanda.benchmarks.downstream_eval import (
    ShortcutDetection,
)
from quanda.benchmarks.resources import config_map
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.datasets.dataset_handlers import get_dataset_handler
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, config, load_from_disk, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist-shortcut-download",
            "load_mnist_unit_test_config",
            False,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            ValueError,
        ),
    ],
)
def test_shortcut_detection(
    test_id,
    config,
    load_from_disk,
    explainer_cls,
    expl_kwargs,
    expected_score,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)

    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }

    config["cache_dir"] = "bench_out"

    if isinstance(expected_score, type):
        with pytest.raises(expected_score):
            dst_eval = ShortcutDetection.from_config(
                config=config,
                load_meta_from_disk=load_from_disk,
            )
        return

    dst_eval = ShortcutDetection.from_config(
        config=config,
        load_meta_from_disk=load_from_disk,
    )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_shortcut_detection",
    ],
)
def test_shortcut_transform_indices(config_name, tmp_path):
    """Verify shortcut-transformed eval samples stay unchanged
    after re-applying the sample transform."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bench = ShortcutDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    ds = bench.eval_dataset
    base_ds = ds.dataset
    transform = ds.dataset_transform
    sample_fn = ds.sample_fn

    total, correct = 0, 0
    for idx in range(len(ds)):
        raw_img, _ = base_ds[idx]
        ds_img, _ = ds[idx]
        once = transform(sample_fn(raw_img))
        if torch.allclose(once, ds_img) == (idx in ds.transform_indices):
            correct += 1
        total += 1

    pct = correct / total
    assert pct == 1.0, (
        f"Expected the transformed samples to differ from the original samples, "
        f"but they were the same for all {total} samples. "
    )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_shortcut_detection",
    ],
)
def test_shortcut_filters(config_name, tmp_path):
    """Verify filter_by_non_shortcut and filter_by_shortcut_pred in benchmark cfg work as expected on eval_dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    bench_yaml = config_map[config_name]

    # Load the benchmark configuration
    with open(bench_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    filter_by_non_shortcut = cfg.get("filter_by_non_shortcut", False)
    filter_by_shortcut_pred = cfg.get("filter_by_shortcut_pred", False)

    bench = ShortcutDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )
    bench.load_last_checkpoint()

    ds_handler = get_dataset_handler(dataset=bench.eval_dataset)
    expl_dl = ds_handler.create_dataloader(
        dataset=bench.eval_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    total, correct = 0, 0
    for i, batch in enumerate(expl_dl):
        inputs, labels = ds_handler.process_batch(
            batch=batch,
            device=device,
        )
        correct_idx = torch.tensor([True] * len(inputs)).to(inputs.device)

        if filter_by_shortcut_pred:
            correct_idx *= labels != bench.shortcut_cls

        if not filter_by_non_shortcut:
            correct += correct_idx.sum().item()
            total += len(inputs)
            continue
        model_inputs = ds_handler.get_model_inputs(inputs=inputs)
        outputs = (
            bench.model(**model_inputs)
            if isinstance(model_inputs, dict)
            else bench.model(model_inputs)
        )
        pred_cls = ds_handler.get_predictions(outputs=outputs)

        correct_idx *= pred_cls == bench.shortcut_cls
        correct += correct_idx.sum().item()
        total += len(pred_cls)

    pct = correct / total
    assert pct == 1.0, (
        f"Expected the filtered samples to match the criteria, "
        f"but {total - correct:.0f}/{total} samples did not match "
        f"(pct={pct:.4f})."
    )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name,expected_thresholds,expected_exact",
    [
        (
            "mnist_shortcut_detection",
            {
                "train_acc": 0.9,
                "val_acc": 0.9,
                "train_shortcut_memorization": 0.85,
                "eval_post_filter_percentage": 0.5,
            },
            {"eval_shortcut_memorization": 1.0},
        ),
        (
            "cifar_shortcut_detection",
            {
                "train_acc": 0.9,
                "val_acc": 0.85,
                "train_shortcut_memorization": 0.85,
                "eval_post_filter_percentage": 0.5,
            },
            {"eval_shortcut_memorization": 1.0},
        ),
    ],
)
def test_shortcut_sanity_check_values(
    config_name, expected_thresholds, expected_exact, tmp_path
):
    """Verify filter_by_non_shortcut and filter_by_shortcut_pred in benchmark cfg work as expected on eval_dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    bench = ShortcutDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    sanity_check_results = bench.sanity_check(batch_size=batch_size)

    for key, threshold in expected_thresholds.items():
        assert sanity_check_results[key] > threshold, (
            f"Expected {key} > {threshold}, "
            f"got {sanity_check_results[key]}."
        )
    for key, value in expected_exact.items():
        assert sanity_check_results[key] == value, (
            f"Expected {key} == {value}, "
            f"got {sanity_check_results[key]}."
        )
