import math
import os

import pytest
import torch

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.downstream_eval import MislabelingDetection
from quanda.benchmarks.downstream_eval.mislabeling_detection import (
    SELF_INFLUENCE_KEY,
)
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.downstream_eval import MislabelingDetectionMetric
from quanda.utils.cache import ExplanationsCache
from quanda.utils.common import _subsample_dataset
from quanda.utils.datasets.transformed import LabelFlippingDataset
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
            0.4991194009780884,
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

    dst_eval = MislabelingDetection.from_config(
        config=config,
        load_meta_from_disk=False,
        offline=True,
        device="cpu",
    )
    train_dataset = dst_eval.train_dataset
    model = dst_eval.model
    checkpoints = dst_eval.checkpoints
    checkpoints_load_func = dst_eval.checkpoints_load_func

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


def _make_bench(config, tmp_path):
    config["cache_dir"] = str(tmp_path)
    return MislabelingDetection.from_config(
        config=config,
        load_meta_from_disk=False,
        offline=True,
        device="cpu",
    )


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "scenario,match",
    [
        ("flipped_eval", "should not have"),
        ("clean_train", "should have"),
        ("length_mismatch", "does not match"),
        ("sanity_not_flipped", None),
        ("subsample_wrong_type", "still be a"),
    ],
)
def test_mislabeling_evaluate(
    scenario, match, load_mnist_mislabeling_config, tmp_path, monkeypatch
):
    """Error/success paths in MislabelingDetection.evaluate and
    sanity_check."""
    config = load_mnist_mislabeling_config
    bench = _make_bench(config, tmp_path)

    if scenario == "flipped_eval":
        bench.eval_dataset = bench.train_dataset
        with pytest.raises(ValueError, match=match):
            bench.evaluate(explainer_cls=CaptumSimilarity)
    elif scenario == "clean_train":
        bench.train_dataset = bench.eval_dataset
        with pytest.raises(ValueError, match=match):
            bench.evaluate(explainer_cls=CaptumSimilarity)
    elif scenario == "length_mismatch":
        cache_dir = str(tmp_path / "bad_cache")
        os.makedirs(cache_dir, exist_ok=True)
        ExplanationsCache.save(
            cache_dir, torch.zeros(3), num_id=SELF_INFLUENCE_KEY
        )
        with pytest.raises(ValueError, match=match):
            bench.evaluate(
                explainer_cls=CaptumSimilarity,
                cache_dir=cache_dir,
                use_cached_expl=True,
            )
    elif scenario == "sanity_not_flipped":
        bench.train_dataset = bench.eval_dataset
        with pytest.raises(TypeError, match="should have flipped"):
            bench.sanity_check(batch_size=8)
    elif scenario == "subsample_wrong_type":
        from quanda.benchmarks.downstream_eval import (
            mislabeling_detection as md,
        )

        monkeypatch.setattr(
            md,
            "_subsample_dataset",
            lambda dataset, max_n, seed: torch.utils.data.Subset(
                dataset, list(range(5))
            ),
        )
        with pytest.raises(TypeError, match=match):
            bench.evaluate(explainer_cls=CaptumSimilarity, max_eval_n=5)

    score = bench.overall_objective(
        {"train_acc": 0.9, "val_acc": 0.9, "mislabeling_memorization": 0.5}
    )
    assert math.isclose(score, 0.1 + 0.2 + 0.35, abs_tol=1e-6)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "sub,match",
    [
        ("happy", None),
        ("wrong_from_config_type", "from_config returned"),
        ("clean_train", "should have"),
    ],
)
def test_mislabeling_explain(
    sub, match, load_mnist_mislabeling_config, tmp_path, monkeypatch
):
    """Error/success paths in MislabelingDetection.explain."""
    config = load_mnist_mislabeling_config
    bench = _make_bench(config, tmp_path)

    expl_kwargs = {
        "layers": "fc_2",
        "similarity_metric": cosine_similarity,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }

    if sub == "wrong_from_config_type":
        monkeypatch.setattr(
            MislabelingDetection,
            "from_config",
            classmethod(lambda cls, config, device="cpu": object()),
        )
    else:
        if sub == "clean_train":
            bench.train_dataset = bench.eval_dataset
        monkeypatch.setattr(
            MislabelingDetection,
            "from_config",
            classmethod(lambda cls, config, device="cpu": bench),
        )

    def fake_self_influence(self, batch_size=8):
        return torch.arange(len(self.train_dataset), dtype=torch.float32)

    monkeypatch.setattr(
        CaptumSimilarity, "self_influence", fake_self_influence
    )

    cache_dir = str(tmp_path / f"expl_{sub}")
    if match is None:
        obj = MislabelingDetection.explain(
            config=config,
            explainer_cls=CaptumSimilarity,
            expl_kwargs=expl_kwargs,
            batch_size=8,
            cache_dir=cache_dir,
            device="cpu",
            max_eval_n=20,
            eval_seed=42,
        )
        assert obj is bench
        assert os.path.exists(
            os.path.join(cache_dir, "explanations_config.yaml")
        )
        assert obj._explanations_dir == cache_dir
    else:
        with pytest.raises(TypeError, match=match):
            MislabelingDetection.explain(
                config=config,
                explainer_cls=CaptumSimilarity,
                expl_kwargs=expl_kwargs,
                cache_dir=cache_dir,
                device="cpu",
            )


@pytest.mark.benchmarks
def test_download_explanations_uses_snapshot(tmp_path, monkeypatch):
    """`Benchmark._download_explanations` mirrors repo_id into cache_dir."""
    calls = {}

    def fake_snapshot_download(repo_id, local_dir, repo_type):
        calls.update(repo_id=repo_id, local_dir=local_dir, repo_type=repo_type)
        os.makedirs(local_dir, exist_ok=True)
        return local_dir

    from quanda.benchmarks import base as bench_base

    monkeypatch.setattr(
        bench_base, "snapshot_download", fake_snapshot_download
    )
    out = Benchmark._download_explanations(
        explanations_id="some-user/some_bench_explanations",
        cache_dir=str(tmp_path),
    )
    expected = os.path.join(
        str(tmp_path),
        "explanations",
        "some-user__some_bench_explanations",
    )
    assert out == expected
    assert calls == {
        "repo_id": "some-user/some_bench_explanations",
        "local_dir": expected,
        "repo_type": "dataset",
    }
    assert os.path.isdir(expected)


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_mislabeling_detection",
        "qnli_mislabeling_detection",
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
    handler = train_ds.handler

    flipped_mismatches = []
    clean_mismatches = []
    for idx in range(len(train_ds)):
        train_label = handler.get_label(train_ds[idx])
        base_label = handler.get_label(base_ds[idx])
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
        "qnli_mislabeling_detection",
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
        (
            "qnli_mislabeling_detection",
            {
                "train_acc": 0.85,
                "val_acc": 0.85,
                "mislabeling_memorization": 0.99,
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
