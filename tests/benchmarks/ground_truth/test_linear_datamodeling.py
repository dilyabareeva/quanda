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
from quanda.benchmarks.ground_truth import (
    linear_datamodeling as lds_module,
)
from quanda.benchmarks.resources import config_map
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

    sanity_results = bench.sanity_check(batch_size=batch_size)

    subset_accs = [
        sanity_results[acc]
        for acc in sanity_results
        if acc.startswith("subset_acc_")
    ]
    assert len(subset_accs) == bench.m, (
        f"Expected {bench.m} subset accuracies, got {len(subset_accs)}."
    )
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
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_linear_datamodeling",
        "cifar_linear_datamodeling",
    ],
)
def test_lds_metadata(
    config_name,
    tmp_path,
    request,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        cfg=cfg, bench_save_dir=str(tmp_path)
    )

    subset_meta = f"{metadata_dir}/{cfg['subset_ids']}"
    with open(subset_meta, "r") as f:
        subset_ids = yaml.safe_load(f)

    assert subset_ids == bench.subset_ids


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

    explanations = torch.randn(
        test_data.shape[0], len(load_mnist_dataset)
    )
    metric.update(
        test_data=test_data,
        explanations=explanations,
        test_targets=test_targets,
    )
    score = metric.compute()["score"]
    assert isinstance(score, float)


@pytest.mark.benchmarks
def test_lds_metric_missing_checkpoints_raises(
    load_mnist_model,
    load_mnist_dataset,
    load_subset_indices_lds,
):
    with open(
        f"tests/assets/lds_checkpoints/{load_subset_indices_lds}", "r"
    ) as f:
        subset_ids = yaml.safe_load(f)

    with pytest.raises(FileNotFoundError):
        LinearDatamodelingMetric(
            model=load_mnist_model,
            train_dataset=load_mnist_dataset,
            alpha=0.5,
            m=1,
            seed=3,
            correlation_fn="spearman",
            subset_ids=subset_ids,
            subset_ckpt_filenames=["/nonexistent/path/model.pt"],
        )


def _make_fake_lds_obj(mocker, m: int = 4):
    """Construct a LinearDatamodeling instance without running training."""
    obj = LinearDatamodeling.__new__(LinearDatamodeling)
    obj.m = m
    obj.train_dataset = mocker.MagicMock()
    obj.model = mocker.MagicMock()
    obj.device = "cpu"
    obj.subset_ids = [[0, 1, 2] for _ in range(m)]
    obj.subset_ckpt_filenames = [
        f"repo/ckpt_lds_subset_{i}" for i in range(m)
    ]
    return obj


@pytest.mark.benchmarks
def test_train_skip_subsets_skips_subset_loop(mocker):
    """With skip_subsets=True, Benchmark.train runs but subset loop is
    skipped."""
    fake_obj = _make_fake_lds_obj(mocker)
    mocker.patch.object(Benchmark, "train", return_value=fake_obj)
    spy = mocker.patch.object(
        LinearDatamodeling, "_train_subset_models"
    )

    config = {"model": {"trainer": {}}, "ckpts": ["repo/ckpt"]}
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
    spy = mocker.patch.object(
        LinearDatamodeling, "_train_subset_models"
    )

    config = {
        "model": {"trainer": {}},
        "ckpts": ["repo/ckpt"],
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
    spy = mocker.patch.object(
        LinearDatamodeling, "_train_subset_model_by_idx"
    )

    config = {
        "model": {"trainer": {}},
        "ckpts": ["repo/ckpt"],
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
    assert kwargs["ckpt_str"] == "repo/ckpt"


@pytest.mark.benchmarks
def test_train_subset_model_by_idx_saves_ckpt(mocker, tmp_path):
    """_train_subset_model_by_idx saves to `<ckpt_dir>_lds_subset_<i>`."""
    obj = _make_fake_lds_obj(mocker, m=2)
    saved_model = mocker.MagicMock()
    mocker.patch.object(
        LinearDatamodelingMetric,
        "train_subset_model",
        return_value=saved_model,
    )

    ckpt_dir = str(tmp_path / "ckpt_base")
    obj._train_subset_model_by_idx(
        i=1,
        trainer=mocker.MagicMock(),
        ckpt_str="repo/ckpt",
        ckpt_dir=ckpt_dir,
        batch_size=4,
        push_to_hub=True,
    )

    expected_dir = f"{ckpt_dir}_lds_subset_1"
    assert os.path.isdir(expected_dir)
    saved_model.save_pretrained.assert_called_once_with(
        expected_dir, safe_serialization=True
    )
    saved_model.push_to_hub.assert_called_once_with(
        "repo/ckpt_lds_subset_1"
    )


@pytest.mark.benchmarks
def test_train_subset_model_by_idx_no_push(mocker, tmp_path):
    """When push_to_hub=False, the hub upload is skipped."""
    obj = _make_fake_lds_obj(mocker, m=2)
    saved_model = mocker.MagicMock()
    mocker.patch.object(
        LinearDatamodelingMetric,
        "train_subset_model",
        return_value=saved_model,
    )

    obj._train_subset_model_by_idx(
        i=0,
        trainer=mocker.MagicMock(),
        ckpt_str="repo/ckpt",
        ckpt_dir=str(tmp_path / "ckpt_base"),
        batch_size=4,
        push_to_hub=False,
    )

    saved_model.push_to_hub.assert_not_called()


@pytest.mark.benchmarks
def test_push_subset_uploads_to_hub(mocker, tmp_path):
    """push_subset calls HfApi.create_repo and upload_folder with the
    subset-postfixed repo id."""
    ckpt_dir = tmp_path / "ckpt" / "repo-ckpt_lds_subset_2"
    ckpt_dir.mkdir(parents=True)

    fake_api = mocker.MagicMock()
    mocker.patch(
        "huggingface_hub.HfApi", return_value=fake_api
    )

    config = {
        "ckpts": ["repo-ckpt"],
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
        yaml.safe_dump({"ckpts": ["repo/ckpt"]}, f)


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
            "--config-path", str(cfg_path),
            "--idx", "5",
            "--device", "cpu",
            "--batch-size", "16",
        ],
    )
    runpy.run_path(
        "scripts/train_lds_subset.py", run_name="__main__"
    )

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
            "--config-path", str(cfg_path),
            "--idx", "7",
            "--push-only",
        ],
    )
    runpy.run_path(
        "scripts/train_lds_subset.py", run_name="__main__"
    )

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
            "--config-path", str(tmp_path / "nope.yaml"),
            "--idx", "0",
        ],
    )
    with pytest.raises(FileNotFoundError):
        runpy.run_path(
            "scripts/train_lds_subset.py", run_name="__main__"
        )
