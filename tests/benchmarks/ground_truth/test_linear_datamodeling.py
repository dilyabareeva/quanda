import os
import runpy
import sys
from copy import deepcopy

import pytest
import torch
import yaml

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.ground_truth import LinearDatamodeling
from quanda.metrics.ground_truth.linear_datamodeling import (
    LinearDatamodelingMetric,
)


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.parametrize(
    "config_name,subset_acc_threshold",
    [
        ("mnist_linear_datamodeling", 0.9),
        ("cifar_linear_datamodeling", 0.7),
        ("qnli_linear_datamodeling", 0.8),
    ],
)
def test_lds_sanity_check_subset_accuracy(
    config_name, subset_acc_threshold, tmp_path
):
    """Verify that all subset checkpoints achieve > threshold accuracy
    on the eval dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    bench = LinearDatamodeling.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )
    bench.subset_ckpt_filenames = bench.subset_ckpt_filenames[:3]

    sanity_results = bench.sanity_check(batch_size=batch_size)

    subset_accs = [
        sanity_results[acc]
        for acc in sanity_results
        if acc.startswith("subset_acc_")
    ]

    for i, acc in enumerate(subset_accs):
        assert acc > subset_acc_threshold, (
            f"Subset checkpoint {i} accuracy {acc:.4f} "
            f"is below the {subset_acc_threshold} threshold."
        )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_linear_datamodeling",
        "cifar_linear_datamodeling",
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
    for ckpt_path in bench.subset_ckpt_filenames[:5]:
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
def test_lds_metadata(load_mnist_linear_datamodeling_config, tmp_path):
    cfg = load_mnist_linear_datamodeling_config

    metadata_dir = BenchConfigParser.get_metadata_dir(
        cfg=cfg, bench_save_dir=str(tmp_path)
    )
    assert metadata_dir == os.path.join(
        str(tmp_path), "metadata", f"{cfg['id']}_metadata"
    )
    assert os.path.isdir(metadata_dir)

    subset_ids = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 0, 1]]
    subset_meta = os.path.join(metadata_dir, cfg["subset_ids"])
    with open(subset_meta, "w") as f:
        yaml.safe_dump(subset_ids, f)

    with open(subset_meta, "r") as f:
        loaded = yaml.safe_load(f)

    assert loaded == subset_ids


@pytest.mark.benchmarks
def test_lds_metric_uses_subset_ckpt_filenames(
    load_mnist_model,
    load_mnist_dataset,
    load_mnist_test_samples_1,
    load_mnist_test_labels_1,
    load_subset_indices_lds,
    load_pretrained_models_lds,
):
    """Verify LDS metric loads subset models lazily from
    subset_ckpt_filenames rather than holding them all in memory."""
    test_data = load_mnist_test_samples_1
    test_targets = torch.tensor(load_mnist_test_labels_1)

    with open(
        f"tests/assets/lds_checkpoints/{load_subset_indices_lds}", "r"
    ) as f:
        subset_ids = yaml.safe_load(f)

    metric = LinearDatamodelingMetric(
        model=load_mnist_model,
        train_dataset=load_mnist_dataset,
        alpha=0.5,
        model_id="mnist_lds",
        m=len(load_pretrained_models_lds),
        seed=3,
        correlation_fn="spearman",
        cache_dir="tests/assets/lds_checkpoints/",
        batch_size=1,
        subset_ids=subset_ids,
        subset_ckpt_filenames=load_pretrained_models_lds,
    )

    assert metric.subset_ckpt_filenames == load_pretrained_models_lds

    explanations = torch.randn(test_data.shape[0], len(load_mnist_dataset))
    metric.update(
        test_data=test_data,
        explanations=explanations,
        test_targets=test_targets,
    )
    score = metric.compute()["score"]
    assert isinstance(score, float)


def _make_fake_lds_obj(mocker, m: int = 4):
    """Construct a LinearDatamodeling instance without running training."""
    obj = LinearDatamodeling.__new__(LinearDatamodeling)
    obj.m = m
    obj.train_dataset = mocker.MagicMock()
    obj.model = mocker.MagicMock()
    obj.device = "cpu"
    obj.subset_ids = [[0, 1, 2] for _ in range(m)]
    obj.subset_ckpt_filenames = [f"repo/ckpt_lds_subset_{i}" for i in range(m)]
    return obj


@pytest.mark.benchmarks
def test_train_skip_subsets_skips_subset_loop(mocker):
    """With skip_subsets=True, Benchmark.train runs but subset loop is
    skipped."""
    fake_obj = _make_fake_lds_obj(mocker)
    mocker.patch.object(Benchmark, "train", return_value=fake_obj)
    spy = mocker.patch.object(LinearDatamodeling, "_train_subset_models")

    config = {"model": {"trainer": {}}, "ckpt": "repo/ckpt"}
    result = LinearDatamodeling.train(config=config, skip_subsets=True)

    assert result is fake_obj
    spy.assert_not_called()


@pytest.mark.benchmarks
def test_train_without_skip_runs_subset_loop(mocker):
    """Without skip_subsets, the subset training loop runs."""
    fake_obj = _make_fake_lds_obj(mocker)
    mocker.patch.object(Benchmark, "train", return_value=fake_obj)
    mocker.patch.object(
        BenchConfigParser, "parse_trainer_cfg", return_value=mocker.MagicMock()
    )
    spy = mocker.patch.object(LinearDatamodeling, "_train_subset_models")

    config = {
        "model": {"trainer": {}},
        "ckpt": "repo/ckpt",
        "bench_save_dir": "/tmp/unused",
        "repo_id": "repo",
    }
    LinearDatamodeling.train(config=config, skip_subsets=False)

    spy.assert_called_once()


@pytest.mark.benchmarks
def test_train_subset_delegates_to_single_idx(mocker):
    """train_subset builds the benchmark and trains only the given idx."""
    fake_obj = _make_fake_lds_obj(mocker)
    mocker.patch.object(
        LinearDatamodeling, "from_config", return_value=fake_obj
    )
    mocker.patch.object(
        BenchConfigParser, "parse_trainer_cfg", return_value=mocker.MagicMock()
    )
    spy = mocker.patch.object(LinearDatamodeling, "_train_subset_model_by_idx")

    config = {
        "model": {"trainer": {}},
        "ckpt": "repo/ckpt",
        "bench_save_dir": "/tmp/unused",
    }
    LinearDatamodeling.train_subset(
        config=config, idx=3, push_to_hub=True, batch_size=16
    )

    spy.assert_called_once()
    kwargs = spy.call_args.kwargs
    assert kwargs["i"] == 3
    assert kwargs["push_to_hub"] is True
    assert kwargs["batch_size"] == 16
    assert kwargs["repo_id"] == "repo/ckpt_lds_subset_3"
    assert kwargs["local_ckpt_dir"] == "/tmp/unused/ckpt/ckpt_lds_subset_3"


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, push_to_hub, pre_populate_dir, expect_warning",
    [
        ("push_empty_dir", True, False, False),
        ("no_push", False, False, False),
        ("push_nonempty_dir", True, True, True),
    ],
)
def test_train_subset_model_by_idx(
    test_id, push_to_hub, pre_populate_dir, expect_warning, mocker, tmp_path
):
    """_train_subset_model_by_idx saves, optionally pushes, and warns
    when the target dir already contains files."""
    import warnings as _warnings

    obj = _make_fake_lds_obj(mocker, m=2)
    saved_model = mocker.MagicMock()
    mocker.patch.object(
        LinearDatamodelingMetric,
        "train_subset_model",
        return_value=saved_model,
    )

    local_ckpt_dir = tmp_path / "ckpt_base_lds_subset_1"
    if pre_populate_dir:
        local_ckpt_dir.mkdir()
        (local_ckpt_dir / "stale.bin").write_text("old")
    repo_id = "repo/ckpt_lds_subset_1"

    with _warnings.catch_warnings(record=True) as rec:
        _warnings.simplefilter("always")
        obj._train_subset_model_by_idx(
            i=1,
            trainer=mocker.MagicMock(),
            local_ckpt_dir=str(local_ckpt_dir),
            repo_id=repo_id,
            batch_size=4,
            push_to_hub=push_to_hub,
        )

    assert os.path.isdir(str(local_ckpt_dir))
    saved_model.save_pretrained.assert_called_once_with(
        str(local_ckpt_dir), safe_serialization=True
    )
    if push_to_hub:
        saved_model.push_to_hub.assert_called_once_with(repo_id)
    else:
        saved_model.push_to_hub.assert_not_called()

    messages = [str(w.message) for w in rec]
    has_warning = any("already exists and is not empty" in m for m in messages)
    assert has_warning is expect_warning


@pytest.mark.benchmarks
def test_push_subset_missing_ckpt_dir_raises(tmp_path):
    """push_subset raises FileNotFoundError for a missing local ckpt dir."""
    config = {"ckpt": "repo-ckpt", "bench_save_dir": str(tmp_path)}
    with pytest.raises(
        FileNotFoundError, match="Subset checkpoint dir missing"
    ):
        LinearDatamodeling.push_subset(config=config, idx=0)


@pytest.mark.benchmarks
def test_extra_kwargs_missing_subset_ids_raises(
    load_mnist_linear_datamodeling_config, tmp_path
):
    """Missing subset_ids file + load_meta_from_disk=True raises."""
    config = load_mnist_linear_datamodeling_config
    metadata_dir = str(tmp_path / "meta")
    os.makedirs(metadata_dir, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Subset ids file not found"):
        LinearDatamodeling._extra_kwargs_from_config(
            config=config,
            train_dataset=torch.utils.data.TensorDataset(
                torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,))
            ),
            eval_dataset=torch.utils.data.TensorDataset(
                torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,))
            ),
            metadata_dir=metadata_dir,
            load_meta_from_disk=True,
        )


@pytest.mark.benchmarks
@pytest.mark.parametrize("skip_subsets", [False, True])
def test_train_and_push_to_hub(mocker, skip_subsets):
    """train_and_push_to_hub toggles the push/skip flags around the base
    call and forwards to Benchmark.train_and_push_to_hub."""
    fake_obj = _make_fake_lds_obj(mocker)
    spy = mocker.patch.object(
        Benchmark, "train_and_push_to_hub", return_value=fake_obj
    )
    config = {
        "model": {"trainer": {}},
        "ckpt": "repo/ckpt",
        "bench_save_dir": "/tmp/unused",
        "skip_subsets": skip_subsets,
    }
    result = LinearDatamodeling.train_and_push_to_hub(config=config)

    assert result is fake_obj
    spy.assert_called_once()
    # Flags must be reset after the call, even when it succeeds.
    assert LinearDatamodeling._push_subsets_during_train is False
    assert LinearDatamodeling._lds_skip_subsets is False


@pytest.mark.benchmarks
def test_train_and_push_to_hub_resets_flags_on_error(mocker):
    """If the base call raises, the class-level flags must still reset."""
    mocker.patch.object(
        Benchmark, "train_and_push_to_hub", side_effect=RuntimeError("boom")
    )
    config = {
        "model": {"trainer": {}},
        "ckpt": "repo/ckpt",
        "bench_save_dir": "/tmp/unused",
        "skip_subsets": True,
    }
    with pytest.raises(RuntimeError):
        LinearDatamodeling.train_and_push_to_hub(config=config)
    assert LinearDatamodeling._push_subsets_during_train is False
    assert LinearDatamodeling._lds_skip_subsets is False


@pytest.mark.benchmarks
def test_push_subset_uploads_to_hub(mocker, tmp_path):
    """push_subset calls HfApi.create_repo and upload_folder with the
    subset-postfixed repo id."""
    ckpt_dir = tmp_path / "ckpt" / "repo-ckpt_lds_subset_2"
    ckpt_dir.mkdir(parents=True)

    fake_api = mocker.MagicMock()
    mocker.patch("huggingface_hub.HfApi", return_value=fake_api)

    config = {
        "ckpt": "repo-ckpt",
        "bench_save_dir": str(tmp_path),
    }
    LinearDatamodeling.push_subset(config=config, idx=2)

    fake_api.create_repo.assert_called_once_with(
        repo_id="repo-ckpt_lds_subset_2", exist_ok=True
    )
    fake_api.upload_folder.assert_called_once_with(
        folder_path=str(ckpt_dir), repo_id="repo-ckpt_lds_subset_2"
    )


def _write_minimal_lds_cfg(path):
    with open(path, "w") as f:
        yaml.safe_dump({"ckpt": "repo/ckpt", "batch_size": 16}, f)


@pytest.mark.benchmarks
def test_train_lds_subset_script_trains(mocker, tmp_path, monkeypatch):
    """scripts/train_lds_subset.py dispatches to LinearDatamodeling.
    train_subset when --push-only is not set."""
    cfg_path = tmp_path / "cfg.yaml"
    _write_minimal_lds_cfg(cfg_path)

    train_spy = mocker.patch.object(LinearDatamodeling, "train_subset")
    push_spy = mocker.patch.object(LinearDatamodeling, "push_subset")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_lds_subset.py",
            "--config-path",
            str(cfg_path),
            "--idx",
            "5",
            "--device",
            "cpu",
        ],
    )
    runpy.run_path("scripts/train_lds_subset.py", run_name="__main__")

    push_spy.assert_not_called()
    train_spy.assert_called_once()
    kwargs = train_spy.call_args.kwargs
    assert kwargs["idx"] == 5
    assert kwargs["batch_size"] == 16
    assert kwargs["push_to_hub"] is False


@pytest.mark.benchmarks
def test_train_lds_subset_script_push_only(mocker, tmp_path, monkeypatch):
    """--push-only dispatches to push_subset without training."""
    cfg_path = tmp_path / "cfg.yaml"
    _write_minimal_lds_cfg(cfg_path)

    train_spy = mocker.patch.object(LinearDatamodeling, "train_subset")
    push_spy = mocker.patch.object(LinearDatamodeling, "push_subset")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_lds_subset.py",
            "--config-path",
            str(cfg_path),
            "--idx",
            "7",
            "--push-only",
        ],
    )
    runpy.run_path("scripts/train_lds_subset.py", run_name="__main__")

    train_spy.assert_not_called()
    push_spy.assert_called_once()
    assert push_spy.call_args.kwargs["idx"] == 7


@pytest.mark.benchmarks
def test_train_lds_subset_script_missing_config(tmp_path, monkeypatch):
    """Script raises FileNotFoundError for a missing config path."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_lds_subset.py",
            "--config-path",
            str(tmp_path / "nope.yaml"),
            "--idx",
            "0",
        ],
    )
    with pytest.raises(FileNotFoundError):
        runpy.run_path("scripts/train_lds_subset.py", run_name="__main__")


@pytest.mark.benchmarks
def test_lds_train_raises_if_super_returns_wrong_type(monkeypatch):
    monkeypatch.setattr(
        Benchmark, "train", classmethod(lambda cls, **kwargs: object())
    )
    monkeypatch.setattr(
        LinearDatamodeling, "_lds_skip_subsets", True, raising=False
    )
    with pytest.raises(
        TypeError, match="Expected a LinearDatamodeling instance"
    ):
        LinearDatamodeling.train(config={})


@pytest.mark.benchmarks
def test_lds_train_subset_raises_if_from_config_returns_wrong_type(
    monkeypatch,
):
    monkeypatch.setattr(
        LinearDatamodeling,
        "from_config",
        classmethod(lambda cls, *args, **kwargs: object()),
    )
    with pytest.raises(
        TypeError, match="Expected a LinearDatamodeling instance"
    ):
        LinearDatamodeling.train_subset(config={}, idx=0)


@pytest.mark.benchmarks
def test_lds_metric_update_uses_subset_logits(mocker):
    """When ``subset_logits`` is supplied, ``update`` skips the
    counterfactual model reload for every subset in the dict."""
    metric = LinearDatamodelingMetric.__new__(LinearDatamodelingMetric)
    metric.m = 3
    metric.subsets = [
        mocker.MagicMock(indices=[0, 1]),
        mocker.MagicMock(indices=[1, 2]),
        mocker.MagicMock(indices=[0, 2]),
    ]
    metric.corr_measure = lambda a, b: torch.zeros(a.shape[0])
    metric.inference_batch_size = None
    metric.results = {"scores": []}

    spy = mocker.patch.object(
        LinearDatamodelingMetric, "load_counterfactual_model"
    )
    subset_logits = {s: torch.randn(4, 2) for s in range(3)}

    metric.update(
        explanations=torch.randn(4, 3),
        test_data=torch.randn(4, 5),
        test_targets=torch.tensor([0, 1, 0, 1]),
        subset_logits=subset_logits,
    )
    spy.assert_not_called()
    assert len(metric.results["scores"]) == 1


@pytest.mark.benchmarks
def test_lds_cache_subset_logits_writes_per_batch_files(mocker, tmp_path):
    """``cache_subset_logits`` writes one ``{i}.pt`` per eval batch,
    each a ``dict[subset_idx -> Tensor]`` across all ``m`` subsets."""
    fake = _make_fake_lds_obj(mocker, m=2)
    fake.eval_dataset = mocker.MagicMock()
    mocker.patch.object(LinearDatamodeling, "from_config", return_value=fake)
    mocker.patch(
        "quanda.benchmarks.ground_truth.linear_datamodeling."
        "_subsample_dataset",
        side_effect=lambda ds, **kw: ds,
    )
    handler = mocker.MagicMock()
    handler.create_dataloader.return_value = ["ba", "bb"]
    handler.process_batch.side_effect = [("ia", None), ("ib", None)]
    mocker.patch(
        "quanda.benchmarks.ground_truth.linear_datamodeling."
        "get_dataset_handler",
        return_value=handler,
    )
    mocker.patch.object(
        LinearDatamodeling,
        "_load_subset_model",
        side_effect=lambda idx, device=None: f"model_{idx}",
    )
    counter = {"n": 0}

    def fake_logits(model, test_data, ibs):
        counter["n"] += 1
        return torch.tensor([[float(counter["n"])]])

    mocker.patch(
        "quanda.benchmarks.ground_truth.linear_datamodeling.chunked_logits",
        side_effect=fake_logits,
    )

    save_dir = LinearDatamodeling.cache_subset_logits(
        config={
            "ckpt": "repo/ckpt",
            "id": "bid",
            "bench_save_dir": str(tmp_path),
        },
        batch_size=4,
        device="cpu",
    )

    assert sorted(f for f in os.listdir(save_dir) if f.endswith(".pt")) == [
        "0.pt",
        "1.pt",
    ]
    for i in range(2):
        d = torch.load(os.path.join(save_dir, f"{i}.pt"))
        assert set(d.keys()) == {0, 1}


@pytest.mark.benchmarks
def test_lds_evaluate_forwards_subset_logits_dir(mocker):
    """``evaluate`` passes ``subset_logits_dir`` through to
    ``_evaluate_dataset``."""
    fake = _make_fake_lds_obj(mocker)
    fake.eval_dataset = mocker.MagicMock()
    fake.checkpoints = ["c"]
    fake.checkpoints_load_func = mocker.MagicMock()
    fake.alpha = 0.5
    fake.cache_dir = "/tmp"
    fake.model_id = "id"
    fake.correlation_fn = lambda a, b: torch.tensor(0.0)
    fake.seed = 0

    mocker.patch.object(
        LinearDatamodeling,
        "_resolve_precomputed_explanations",
        return_value=None,
    )
    mocker.patch.object(
        LinearDatamodeling,
        "_prepare_explainer",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "quanda.benchmarks.ground_truth.linear_datamodeling."
        "LinearDatamodelingMetric",
        return_value=mocker.MagicMock(),
    )
    spy = mocker.patch.object(
        LinearDatamodeling,
        "_evaluate_dataset",
        return_value={"score": 0.0},
    )

    LinearDatamodeling.evaluate(
        fake,
        explainer_cls=type("StubExplainer", (), {}),
        subset_logits_dir="/my/logits",
    )
    assert spy.call_args.kwargs["subset_logits_dir"] == "/my/logits"


@pytest.mark.benchmarks
def test_lds_metric_update_partial_subset_logits(mocker):
    metric = LinearDatamodelingMetric.__new__(LinearDatamodelingMetric)
    metric.m = 3
    metric.subsets = [mocker.MagicMock(indices=[0, 1]) for _ in range(3)]
    metric.corr_measure = lambda a, b: torch.zeros(a.shape[0])
    metric.inference_batch_size = None
    metric.results = {"scores": []}

    load_spy = mocker.patch.object(
        LinearDatamodelingMetric,
        "load_counterfactual_model",
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "quanda.metrics.ground_truth.linear_datamodeling.chunked_logits",
        return_value=torch.randn(4, 2),
    )

    metric.update(
        explanations=torch.randn(4, 3),
        test_data=torch.randn(4, 5),
        test_targets=torch.tensor([0, 1, 0, 1]),
        subset_logits={0: torch.randn(4, 2)},
    )
    # Only subset 0 was provided: subsets 1 and 2 must be loaded.
    assert load_spy.call_count == 2
    loaded_idxs = sorted(c.args[0] for c in load_spy.call_args_list)
    assert loaded_idxs == [1, 2]


@pytest.mark.benchmarks
def test_subset_logits_cache_dir_is_deterministic():

    config = {
        "ckpt": "repo/ckpt",
        "id": "bid",
        "bench_save_dir": "/tmp",
    }
    base = LinearDatamodeling.subset_logits_cache_dir(
        config=config, batch_size=8, max_eval_n=1000, eval_seed=42
    )
    assert (
        LinearDatamodeling.subset_logits_cache_dir(
            config=config, batch_size=8, max_eval_n=1000, eval_seed=42
        )
        == base
    )
    for bs, n, s in [(16, 1000, 42), (8, 500, 42), (8, 1000, 7)]:
        assert (
            LinearDatamodeling.subset_logits_cache_dir(
                config=config, batch_size=bs, max_eval_n=n, eval_seed=s
            )
            != base
        )


@pytest.mark.benchmarks
def test_evaluate_dataset_without_subset_logits_dir_skips_injection(mocker):

    fake = _make_fake_lds_obj(mocker)
    mocker.patch.object(
        LinearDatamodeling,
        "_iter_explanations",
        side_effect=lambda *a, **kw: iter(
            [(0, "inputs", "labels", "targets", "expl", 1)]
        ),
    )
    metric = mocker.MagicMock()
    metric.compute.return_value = {"score": 0.0}

    LinearDatamodeling._evaluate_dataset(
        fake,
        eval_dataset=mocker.MagicMock(),
        explainer=None,
        metric=metric,
        batch_size=1,
    )
    assert "subset_logits" not in metric.update.call_args.kwargs


@pytest.mark.benchmarks
def test_evaluate_dataset_loads_subset_logits_from_dir(mocker, tmp_path):

    fake = _make_fake_lds_obj(mocker)
    torch.save(
        {0: torch.tensor([1.0, 2.0])},
        os.path.join(tmp_path, "0.pt"),
    )

    mocker.patch.object(
        LinearDatamodeling,
        "_iter_explanations",
        side_effect=lambda *a, **kw: iter(
            [(0, "inputs", "labels", "targets", "expl", 1)]
        ),
    )
    metric = mocker.MagicMock()
    metric.compute.return_value = {"score": 0.0}

    LinearDatamodeling._evaluate_dataset(
        fake,
        eval_dataset=mocker.MagicMock(),
        explainer=None,
        metric=metric,
        batch_size=1,
        subset_logits_dir=str(tmp_path),
    )
    kwargs = metric.update.call_args.kwargs
    assert "subset_logits" in kwargs
    assert torch.equal(kwargs["subset_logits"][0], torch.tensor([1.0, 2.0]))


@pytest.mark.benchmarks
def test_evaluate_dataset_skips_missing_subset_logits_file(mocker, tmp_path):

    fake = _make_fake_lds_obj(mocker)

    mocker.patch.object(
        LinearDatamodeling,
        "_iter_explanations",
        side_effect=lambda *a, **kw: iter(
            [(0, "inputs", "labels", "targets", "expl", 1)]
        ),
    )
    metric = mocker.MagicMock()
    metric.compute.return_value = {"score": 0.0}

    LinearDatamodeling._evaluate_dataset(
        fake,
        eval_dataset=mocker.MagicMock(),
        explainer=None,
        metric=metric,
        batch_size=1,
        subset_logits_dir=str(tmp_path),
    )
    assert "subset_logits" not in metric.update.call_args.kwargs
