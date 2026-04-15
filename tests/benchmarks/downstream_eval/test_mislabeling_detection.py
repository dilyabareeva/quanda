import math
import os

import pytest
import torch

from quanda.benchmarks.base import _subsample_dataset
from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.downstream_eval import MislabelingDetection
from quanda.benchmarks.downstream_eval.mislabeling_detection import (
    SELF_INFLUENCE_KEY,
)
from quanda.benchmarks.resources.sample_transforms import sample_transforms
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.downstream_eval import MislabelingDetectionMetric
from quanda.utils.cache import ExplanationsCache
from quanda.utils.datasets.transformed import LabelFlippingDataset
from quanda.utils.datasets.transformed.metadata import LabelFlippingMetadata
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, config, global_method, load_from_disk,"
    "explainer_cls, expl_kwargs, max_eval_n, eval_seed,"
    "mock_self_influence, use_cached_expl, expected_score",
    [
        (
            "mnist",
            "load_mnist_mislabeling_config",
            "self-influence",
            False,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            None,
            42,
            False,
            False,
            0.44353821873664856,
        ),
        (
            "mnist_subset_cached_mocked",
            "load_mnist_mislabeling_config",
            "self-influence",
            False,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            50,
            42,
            True,
            True,
            None,
        ),
    ],
)
def test_mislabeling_detection(
    test_id,
    config,
    global_method,
    load_from_disk,
    explainer_cls,
    expl_kwargs,
    max_eval_n,
    eval_seed,
    mock_self_influence,
    use_cached_expl,
    expected_score,
    tmp_path,
    request,
    monkeypatch,
):
    config = request.getfixturevalue(config)

    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }

    config["cache_dir"] = str(tmp_path)

    train_metadata = LabelFlippingMetadata(
        p=config["train_dataset"]["wrapper"]["metadata"]["p"],
        seed=config["train_dataset"]["wrapper"]["metadata"]["seed"],
    )
    train_dataset = LabelFlippingDataset(
        dataset=BenchConfigParser._parse_hf_dataset(
            dataset=config["train_dataset"]["dataset_str"],
            transform=sample_transforms[config["train_dataset"]["transforms"]],
            dataset_split=config["train_dataset"]["dataset_split"],
        ),
        metadata=train_metadata,
    )

    eval_dataset = BenchConfigParser._parse_hf_dataset(
        dataset=config["eval_dataset"]["dataset_str"],
        transform=sample_transforms[config["eval_dataset"]["transforms"]],
        dataset_split=config["eval_dataset"]["dataset_split"],
    )

    model, checkpoints, checkpoints_load_func = (
        BenchConfigParser.parse_model_cfg(
            config["model"],
            config["bench_save_dir"],
            config["ckpts"],
            load_model_from_disk=True,
            device="cpu",
        )
    )
    dst_eval = MislabelingDetection(
        train_dataset=train_dataset,
        device="cpu",
        eval_dataset=eval_dataset,
        model=model,
        checkpoints=checkpoints,
        checkpoints_load_func=checkpoints_load_func,
    )

    cache_dir = None
    if mock_self_influence:
        # Replace the explainer's self_influence with a deterministic stub
        # so the test doesn't depend on real attribution computation.
        def fake_self_influence(self, batch_size=8):
            return torch.arange(len(self.train_dataset), dtype=torch.float32)

        monkeypatch.setattr(
            CaptumSimilarity, "self_influence", fake_self_influence
        )

    if use_cached_expl:
        # Pre-write the cache exactly as MislabelingDetection.explain would,
        # so evaluate(use_cached_expl=True) loads it from disk.
        cache_dir = str(tmp_path / "expl_cache")
        os.makedirs(cache_dir, exist_ok=True)
        train_subset = _subsample_dataset(
            train_dataset, max_n=max_eval_n, seed=eval_seed
        )
        precomputed_si = torch.arange(len(train_subset), dtype=torch.float32)
        ExplanationsCache.save(
            cache_dir, precomputed_si, num_id=SELF_INFLUENCE_KEY
        )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
        max_eval_n=max_eval_n,
        eval_seed=eval_seed,
        cache_dir=cache_dir,
        use_cached_expl=use_cached_expl,
    )["score"]

    if expected_score is None:
        # No hardcoded reference: assert equivalence with a direct
        # MislabelingDetectionMetric call using the same precomputed
        # tensor + remapped indices. This pins down the cache-load +
        # subset-remap path without depending on a magic number.
        train_subset = _subsample_dataset(
            train_dataset, max_n=max_eval_n, seed=eval_seed
        )
        reference_si = torch.arange(len(train_subset), dtype=torch.float32)
        reference_metric = MislabelingDetectionMetric(
            model=model,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            train_dataset=train_subset,
            mislabeling_indices=train_subset.transform_indices,
            precomputed_self_influence=reference_si,
        )
        expected_score = reference_metric.compute()["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_mislabeling_detection",
    ],
)
def test_train_dataset_mislabeling_is_correct(config_name, tmp_path):
    """Verify that label flipping in the train dataset is applied
    correctly: transformed indices have flipped labels and
    non-transformed indices keep their original labels."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bench = MislabelingDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    train_ds = bench.train_dataset
    assert isinstance(train_ds, LabelFlippingDataset), (
        "Expected train_dataset to be a LabelFlippingDataset, "
        f"got {type(train_ds).__name__}."
    )

    base_ds = train_ds.dataset
    transform_indices = set(train_ds.transform_indices)

    flipped_mismatches = []
    clean_mismatches = []
    for idx in range(len(train_ds)):
        _, train_label = train_ds[idx]
        _, base_label = base_ds[idx]
        if idx in transform_indices:
            if train_label == base_label:
                flipped_mismatches.append(idx)
        else:
            if train_label != base_label:
                clean_mismatches.append(idx)

    assert not flipped_mismatches, (
        f"{len(flipped_mismatches)} samples at transform_indices "
        f"still have their original label (expected flipped)."
    )
    assert not clean_mismatches, (
        f"{len(clean_mismatches)} non-transformed samples have a "
        f"different label than the base dataset (expected unchanged)."
    )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_mislabeling_detection",
    ],
)
def test_eval_dataset_is_clean(config_name, tmp_path):
    """Verify the eval dataset is NOT a LabelFlippingDataset,
    i.e. it contains no mislabeled samples."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bench = MislabelingDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    assert not isinstance(bench.eval_dataset, LabelFlippingDataset), (
        "Eval dataset should be clean (no label flipping), "
        f"but got {type(bench.eval_dataset).__name__}."
    )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name,expected_thresholds",
    [
        (
            "mnist_mislabeling_detection",
            {
                "train_acc": 0.85,
                "val_acc": 0.85,
                "mislabeling_memorization": 0.4,
            },
        ),
        (
            "cifar_mislabeling_detection",
            {
                "train_acc": 0.85,
                "val_acc": 0.8,
                "mislabeling_memorization": 0.4,
            },
        ),
    ],
)
def test_mislabeling_sanity_check_values(
    config_name, expected_thresholds, tmp_path
):
    """Verify model fitness: train/val accuracy and mislabeling
    memorization are within expected bounds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    bench = MislabelingDetection.load_pretrained(
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
