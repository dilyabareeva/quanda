"""Tests for the Benchmark explanation-cache plumbing."""

import math
import os
from unittest import mock

import pytest
import torch
import yaml

from quanda.benchmarks.base import (
    _hash_expl_kwargs,
    _stable_repr,
    default_explanations_id,
)
from quanda.benchmarks.downstream_eval import ClassDetection
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.cache import BatchedCachedExplanations, ExplanationsCache
from quanda.utils.functions import cosine_similarity


def test_hash_expl_kwargs_is_order_invariant():
    a = _hash_expl_kwargs({"a": 1, "b": 2})
    b = _hash_expl_kwargs({"b": 2, "a": 1})
    c = _hash_expl_kwargs({"a": 1, "b": 3})
    assert a == b
    assert a != c
    assert _hash_expl_kwargs(None) == _hash_expl_kwargs({})


def test_hash_expl_kwargs_is_stable_for_callables():
    """Callables must hash by module.qualname, not ``id(obj)``."""
    rep = _stable_repr(cosine_similarity)
    assert "at 0x" not in rep
    assert rep == (
        f"{cosine_similarity.__module__}.{cosine_similarity.__qualname__}"
    )
    # Two distinct function objects with the same qualname must hash equally.
    h1 = _hash_expl_kwargs({"f": cosine_similarity})

    def _wrap(*a, **kw):
        return cosine_similarity(*a, **kw)

    _wrap.__module__ = cosine_similarity.__module__
    _wrap.__qualname__ = cosine_similarity.__qualname__
    h2 = _hash_expl_kwargs({"f": _wrap})
    assert h1 == h2


def test_default_explanations_id_format():
    cfg = {"id": "bench-x", "repo_id": "owner"}
    out = default_explanations_id(
        cfg, CaptumSimilarity, {"k": 1}, max_eval_n=1000, eval_seed=42
    )
    h = _hash_expl_kwargs({"k": 1})
    assert out == (
        f"owner/bench-x__CaptumSimilarity__{h}__n1000_s42_explanations"
    )

    cfg2 = {"id": "bench-y"}
    assert default_explanations_id(cfg2, CaptumSimilarity, None).startswith(
        "quanda-bench-test/bench-y__CaptumSimilarity__"
    )


def test_batched_cache_indexed_by_num_id(tmp_path):
    path = str(tmp_path)
    for i in [2, 0, 10, 1]:
        ExplanationsCache.save(path, torch.full((2, 3), float(i)), num_id=i)
    bc = BatchedCachedExplanations(cache_dir=path, device="cpu")
    assert len(bc) == 4
    assert set(bc.keys()) == {0, 1, 2, 10}
    assert torch.equal(bc[10], torch.full((2, 3), 10.0))
    assert torch.equal(bc[0], torch.full((2, 3), 0.0))
    with pytest.raises(KeyError):
        _ = bc[42]

    ExplanationsCache.save(path, torch.zeros(2, 3), num_id="alpha")
    bc2 = BatchedCachedExplanations(cache_dir=path, device="cpu")
    assert "alpha" in bc2.keys()


@pytest.mark.benchmarks
def test_benchmark_explain_and_precomputed_evaluate_match(
    load_mnist_unit_test_config, tmp_path
):
    """End-to-end: explain → load cache → evaluate matches direct evaluate."""
    config = load_mnist_unit_test_config
    config["cache_dir"] = "bench_out"

    expl_kwargs = {
        "layers": "fc_2",
        "similarity_metric": cosine_similarity,
        "model_id": "test",
        "cache_dir": str(tmp_path / "expl_cache"),
    }

    direct = ClassDetection.from_config(
        config=config, load_meta_from_disk=True, offline=True
    )
    baseline = direct.evaluate(
        explainer_cls=CaptumSimilarity,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    explanations_id = "owner/test_explanations"
    cache_root = str(tmp_path / "expl_root")
    explanations_dir = os.path.join(
        cache_root, "explanations", explanations_id.replace("/", "__")
    )
    obj = ClassDetection.explain(
        config=config,
        explainer_cls=CaptumSimilarity,
        expl_kwargs=expl_kwargs,
        batch_size=8,
        explanations_id=explanations_id,
        cache_dir=explanations_dir,
        device="cpu",
    )
    assert obj._explanations_id == explanations_id
    assert obj._explanations_dir == explanations_dir
    meta_path = os.path.join(explanations_dir, "explanations_config.yaml")
    assert os.path.exists(meta_path)
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    assert meta["explainer_cls"] == "CaptumSimilarity"
    assert meta["explanations_id"] == explanations_id
    assert meta["bench"] == config.get("bench")
    pt_files = [p for p in os.listdir(explanations_dir) if p.endswith(".pt")]
    assert len(pt_files) == meta["n_batches"]

    fresh = ClassDetection.from_config(
        config=config,
        load_meta_from_disk=True,
        offline=True,
    )
    cached_score = fresh.evaluate(
        explainer_cls=CaptumSimilarity,
        expl_kwargs=expl_kwargs,
        batch_size=8,
        cache_dir=explanations_dir,
        use_cached_expl=True,
    )["score"]
    assert math.isclose(cached_score, baseline, abs_tol=1e-6)

    # No-op when both flags are False: runs the explainer and matches too.
    noop_score = fresh.evaluate(
        explainer_cls=CaptumSimilarity,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]
    assert math.isclose(noop_score, baseline, abs_tol=1e-6)

    # Missing cache_dir when flags are set should raise.
    with pytest.raises(ValueError, match="cache_dir must be provided"):
        fresh.evaluate(
            explainer_cls=CaptumSimilarity,
            expl_kwargs=expl_kwargs,
            batch_size=8,
            use_cached_expl=True,
        )

    # use_hf_expl: mock snapshot_download to populate the target dir from
    # the existing local cache, then verify the load path matches.
    hf_dir = str(tmp_path / "hf_cache" / "owner__test_explanations")

    def fake_snapshot_download(repo_id, local_dir, repo_type):
        assert repo_id == "owner/test_explanations"
        assert repo_type == "dataset"
        os.makedirs(local_dir, exist_ok=True)
        for name in os.listdir(explanations_dir):
            with open(os.path.join(explanations_dir, name), "rb") as src:
                with open(os.path.join(local_dir, name), "wb") as dst:
                    dst.write(src.read())

    with mock.patch(
        "quanda.benchmarks.base.snapshot_download",
        side_effect=fake_snapshot_download,
    ):
        hf_score = fresh.evaluate(
            explainer_cls=CaptumSimilarity,
            expl_kwargs=expl_kwargs,
            batch_size=8,
            cache_dir=hf_dir,
            use_hf_expl=True,
        )["score"]
    assert math.isclose(hf_score, baseline, abs_tol=1e-6)
