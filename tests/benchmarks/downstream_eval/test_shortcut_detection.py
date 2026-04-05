import math

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


#@pytest.mark.production_bench
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
    assert pct == 1., (
        f"Expected the transformed samples to differ from the original samples, "
        f"but they were the same for all {total} samples. "
    )


#@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_shortcut_detection",
    ],
)
def test_shortcut_filters(config_name, tmp_path):
    """Verify filter_by_non_shortcut and filter_by_class in benchmark cfg work as expected on eval_dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    bench_yaml = config_map[config_name]

    # Load the benchmark configuration
    with open(bench_yaml, "r") as f:
        cfg = yaml.safe_load(f)
            
    filter_by_non_shortcut = cfg.get("filter_by_non_shortcut", False)
    filter_by_class = cfg.get("filter_by_class", False)
    
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

        if filter_by_class:
            correct_idx *= labels != bench.shortcut_cls
            
        if not filter_by_non_shortcut:
            correct += correct_idx.sum().item()
            total += len(pred_cls)
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
    assert pct == 1., (
        f"Expected the filtered samples to match the criteria, "
        f"but {total - correct:.0f}/{total} samples did not match "
        f"(pct={pct:.4f})."
    )


#@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_shortcut_detection",
    ],
)
def test_shortcut_sanity_check_values(config_name, tmp_path):
    """Verify filter_by_non_shortcut and filter_by_class in benchmark cfg work as expected on eval_dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    bench_yaml = config_map[config_name]
    with open(bench_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    bench = ShortcutDetection.load_pretrained(
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
    
    assert( actual_ratio / eval_ratio) > 0.5, (
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
    assert sanity_check_results["train_shortcut_memorization"] > 0.77, (
        f"Expected train_shortcut_memorization to be > 0.77, but got {sanity_check_results['train_shortcut_memorization']}."
    )
    assert sanity_check_results["eval_shortcut_memorization"] > 0.9, (
        f"Expected eval_shortcut_memorization to be > 0.9, but got {sanity_check_results['eval_shortcut_memorization']}."
    )
    