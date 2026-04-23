"""Contains tests for parsing Hydra/yaml benchmark configs."""

import math
import os

import datasets as hf_datasets
import pytest
import torch

from quanda.benchmarks import config_parser as cp_module
from quanda.benchmarks.config_parser import BenchConfigParser


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, config, input_shape",
    [
        (
            "mnist",
            "load_mnist_unit_test_config_hf",
            (1, 28, 28),
        ),
    ],
)
def test_load_ckpt_from_hf(
    test_id,
    config,
    input_shape,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)

    rand_input = torch.rand(1, *input_shape)

    model, ckpt, load_fn = BenchConfigParser.parse_model_cfg(
        config["model"],
        str(tmp_path),
        [config["ckpt"]],
        False,
        "cpu",
    )
    load_fn(model, ckpt[-1])
    out_offline = model(rand_input).mean().item()

    model, ckpt, load_fn = BenchConfigParser.parse_model_cfg(
        config["model"],
        str(tmp_path),
        [config["ckpt"]],
        True,
        "cpu",
    )
    load_fn(model, ckpt[-1])
    out_online = model(rand_input).mean().item()

    assert math.isclose(out_offline, out_online, rel_tol=1e-5)


@pytest.mark.utils
def test_load_metadata_offline_missing_dir_raises(tmp_path):
    """offline=True with a missing metadata dir must raise FileNotFoundError."""
    missing_dir = str(tmp_path / "does_not_exist")
    with pytest.raises(FileNotFoundError, match="Metadata directory"):
        BenchConfigParser.load_metadata(
            cfg={"id": "x", "repo_id": "y"},
            metadata_dir=missing_dir,
            offline=True,
        )


@pytest.mark.utils
def test_parse_model_cfg_offline_missing_ckpt_raises(
    load_mnist_unit_test_config_hf, tmp_path
):
    """offline=True with no local checkpoint must raise FileNotFoundError."""
    config = load_mnist_unit_test_config_hf

    _, ckpt_ids, load_fn = BenchConfigParser.parse_model_cfg(
        model_cfg=config["model"],
        bench_save_dir=str(tmp_path),
        ckpts=[config["ckpt"]],
        offline=True,
        device="cpu",
    )

    with pytest.raises(FileNotFoundError, match="offline=True"):
        load_fn(torch.nn.Linear(1, 1), ckpt_ids[-1])


@pytest.mark.utils
def test_parse_model_cfg_load_state_dict_failure_raises(
    load_mnist_unit_test_config_hf, tmp_path
):
    """Corrupt local checkpoint must surface as a ValueError."""
    config = load_mnist_unit_test_config_hf

    model, ckpt_ids, load_fn = BenchConfigParser.parse_model_cfg(
        model_cfg=config["model"],
        bench_save_dir=str(tmp_path),
        ckpts=[config["ckpt"]],
        offline=False,
        device="cpu",
    )
    ckpt_name = config["ckpt"].split("/")[-1]
    ckpt_dir = os.path.join(str(tmp_path), "ckpt", ckpt_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        f.write("not valid json")

    with pytest.raises(ValueError, match="Error loading model from"):
        load_fn(model, ckpt_ids[-1])


@pytest.mark.utils
def test_load_dataset_from_cfg_false_single_class_raises(tmp_path):
    """single_class_dataset=False in ds_config triggers the catch-all raise."""
    ds_config = {"single_class_dataset": False}
    with pytest.raises(
        ValueError, match="Dataset configuration not recognized"
    ):
        BenchConfigParser._load_dataset_from_cfg(
            ds_config=ds_config,
            metadata_dir=str(tmp_path),
        )


@pytest.mark.utils
def test_apply_indices_with_hf_dataset(tmp_path):
    """HF dataset branch in _apply_indices uses .select(indices)."""
    hf_ds = hf_datasets.Dataset.from_dict(
        {"x": [0, 1, 2, 3, 4], "label": [0, 1, 0, 1, 0]}
    )
    result = BenchConfigParser._apply_indices(
        base_dataset=hf_ds,
        ds_config={},
        metadata_dir=str(tmp_path),
    )
    assert isinstance(result, hf_datasets.Dataset)
    assert len(result) == 5


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, ref, splits_cfg, exc, match",
    [
        ("missing_ref", "unknown", {}, KeyError, "split_ref 'unknown'"),
        (
            "incomplete_recipe",
            "mnist_train",
            {"mnist_train": {"filename": "x.yaml"}},
            ValueError,
            "must define 'filename' and 'ratios'",
        ),
    ],
)
def test_resolve_split_recipe_raises(test_id, ref, splits_cfg, exc, match):
    """_resolve_split_recipe rejects missing refs and incomplete recipes."""
    with pytest.raises(exc, match=match):
        BenchConfigParser._resolve_split_recipe(ref, splits_cfg)


@pytest.mark.utils
def test_resolve_split_recipe_returns_copy():
    """A fully-formed recipe is returned (as a deepcopy)."""
    recipe = {"filename": "x.yaml", "ratios": {"train": 0.9, "test": 0.1}}
    result = BenchConfigParser._resolve_split_recipe(
        "mnist_train", {"mnist_train": recipe}
    )
    assert result == recipe
    assert result is not recipe


@pytest.mark.utils
def test_apply_wrapper_missing_metadata_raises(
    load_mnist_mislabeling_config, tmp_path
):
    """load_meta_from_disk=True with missing metadata file raises."""
    config = load_mnist_mislabeling_config
    metadata_dir = str(tmp_path / "meta")
    os.makedirs(metadata_dir, exist_ok=True)

    dummy_ds = torch.utils.data.TensorDataset(
        torch.randn(4, 1, 28, 28), torch.randint(0, 10, (4,))
    )

    ds_cfg = config["train_dataset"]
    wrapper_cfg = dict(ds_cfg["wrapper"])
    wrapper_cfg["metadata"] = {
        **wrapper_cfg["metadata"],
        "metadata_filename": "not_there.yaml",
    }

    with pytest.raises(FileNotFoundError, match="Wrapper metadata"):
        BenchConfigParser._apply_wrapper(
            dataset=dummy_ds,
            ds_config=ds_cfg,
            wrapper_cfg=wrapper_cfg,
            metadata_dir=metadata_dir,
            load_meta_from_disk=True,
        )


@pytest.mark.utils
def test_load_pretrained_base_returns_none_when_key_absent():
    """Without ``pretrained_model_name`` in the cfg, ``load_pretrained_base``
    must short-circuit to ``None`` so train paths keep the empty-architecture
    model produced by ``parse_model_cfg``."""
    cfg = {"module": {"name": "MnistTorch", "args": {}}}
    assert BenchConfigParser.load_pretrained_base(cfg, device="cpu") is None


@pytest.mark.utils
def test_load_pretrained_base_invokes_from_pretrained_base(monkeypatch):
    """The happy path routes through
    ``module_cls.from_pretrained_base(pretrained_model_name=...)`` and
    ``.to(device)``."""
    calls = {}

    class _FakeModule(torch.nn.Linear):
        def __init__(self):
            super().__init__(1, 1)

        @classmethod
        def from_pretrained_base(cls, pretrained_model_name, num_labels):
            calls["name"] = pretrained_model_name
            calls["num_labels"] = num_labels
            return cls()

    monkeypatch.setitem(cp_module.pl_modules, "FakeForPretrained", _FakeModule)
    cfg = {
        "pretrained_model_name": "fake/base",
        "num_labels": 3,
        "module": {"name": "FakeForPretrained", "args": {}},
    }
    model = BenchConfigParser.load_pretrained_base(cfg, device="cpu")
    assert isinstance(model, _FakeModule)
    assert calls["name"] == "fake/base"
    assert calls["num_labels"] == 3


@pytest.mark.utils
def test_parse_model_cfg_rejects_non_module_instance(monkeypatch, tmp_path):
    """If the registered class doesn't return ``torch.nn.Module``, the parser
    raises early rather than silently proceeding."""

    class _NotAModule:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setitem(cp_module.pl_modules, "NotAModule", _NotAModule)

    cfg = {
        "module": {"name": "NotAModule", "args": {}},
        "trainer": {"lr": 0.01},
    }
    with pytest.raises(ValueError, match="did not return a"):
        BenchConfigParser.parse_model_cfg(
            model_cfg=cfg,
            bench_save_dir=str(tmp_path),
            ckpts=["repo/any"],
            offline=True,
            device="cpu",
        )
