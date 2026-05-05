"""Contains tests for parsing Hydra/yaml benchmark configs."""

import math
import os

import datasets as hf_datasets
import pytest
import torch

from quanda.benchmarks import config_parser as cp_module
from quanda.benchmarks.config_parser import (
    DatasetConfigParser,
    FactTracingConfigParser,
    MetadataConfigParser,
    ModelConfigParser,
)


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

    model, ckpt, load_fn = ModelConfigParser.parse_model_cfg(
        config["model"],
        str(tmp_path),
        [config["ckpt"]],
        False,
        "cpu",
    )
    load_fn(model, ckpt[-1])
    out_offline = model(rand_input).mean().item()

    model, ckpt, load_fn = ModelConfigParser.parse_model_cfg(
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
        MetadataConfigParser.load_metadata(
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

    _, ckpt_ids, load_fn = ModelConfigParser.parse_model_cfg(
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

    model, ckpt_ids, load_fn = ModelConfigParser.parse_model_cfg(
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
        DatasetConfigParser._load_dataset_from_cfg(
            ds_config=ds_config,
            metadata_dir=str(tmp_path),
        )


@pytest.mark.utils
def test_apply_indices_with_hf_dataset(tmp_path):
    """HF dataset branch in _apply_indices uses .select(indices)."""
    hf_ds = hf_datasets.Dataset.from_dict(
        {"x": [0, 1, 2, 3, 4], "label": [0, 1, 0, 1, 0]}
    )
    result = DatasetConfigParser._apply_indices(
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
        DatasetConfigParser._resolve_split_recipe(ref, splits_cfg)


@pytest.mark.utils
def test_resolve_split_recipe_returns_copy():
    """A fully-formed recipe is returned (as a deepcopy)."""
    recipe = {"filename": "x.yaml", "ratios": {"train": 0.9, "test": 0.1}}
    result = DatasetConfigParser._resolve_split_recipe(
        "mnist_train", {"mnist_train": recipe}
    )
    assert result == recipe
    assert result is not recipe


@pytest.mark.utils
def test_load_pretrained_base_returns_none_when_key_absent():
    """Without ``pretrained_model_name`` in the cfg, ``load_pretrained_base``
    must short-circuit to ``None`` so train paths keep the empty-architecture
    model produced by ``parse_model_cfg``."""
    cfg = {"module": {"name": "MnistTorch", "args": {}}}
    assert ModelConfigParser.load_pretrained_base(cfg, device="cpu") is None


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
    model = ModelConfigParser.load_pretrained_base(cfg, device="cpu")
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
        ModelConfigParser.parse_model_cfg(
            model_cfg=cfg,
            bench_save_dir=str(tmp_path),
            ckpts=["repo/any"],
            offline=True,
            device="cpu",
        )


class _FakeTokenizer:
    """Minimal tokenize() callable for FactTracingConfigParser tests.

    Returns a token id per character (so ``len(input_ids)`` is
    deterministic from the input length) plus the standard
    padding/truncation behaviour.
    """

    pad_token_id = 0

    def __call__(self, text, padding=False, truncation=False, max_length=None):
        ids = [ord(c) % 100 + 1 for c in text]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length" and max_length is not None:
            pad = max_length - len(ids)
            return {
                "input_ids": ids + [self.pad_token_id] * pad,
                "attention_mask": [1] * len(ids) + [0] * pad,
            }
        return {"input_ids": list(ids), "attention_mask": [1] * len(ids)}


def _fake_hf_dataset(rows):
    return hf_datasets.Dataset.from_list(rows)


@pytest.mark.utils
def test_parse_fact_tracing_cfg_basic(monkeypatch):
    """Builds prompt/evidence datasets, entailment matrix, and pad id."""
    rows = [
        {
            "prompt": "Q1?",
            "answer": ["A1"],
            "evidence_sentences": ["E1a", "E1b", "E1c"],
        },
        {
            "prompt": "Q2?",
            "answer": ["A2"],
            "evidence_sentences": ["E2a", "E2b"],
        },
    ]
    monkeypatch.setattr(
        cp_module, "load_dataset", lambda *a, **kw: _fake_hf_dataset(rows)
    )
    fake_tok = _FakeTokenizer()
    monkeypatch.setattr(
        cp_module, "resolve_tokenizer", lambda cfg: (fake_tok, 0)
    )

    cfg = {
        "dataset_str": "fake/ds",
        "tokenizer": {"backend": "hf", "name": "irrelevant"},
        "num_prompts": 5,  # >= len(rows) → no sampling
        "max_length": 16,
        "max_evidence_per_prompt": 2,
    }

    prompt_ds, evidence_ds, labels, pad_id = (
        FactTracingConfigParser.parse_fact_tracing_cfg(cfg)
    )

    assert pad_id == 0
    assert len(prompt_ds) == 2
    assert prompt_ds["prompt"] == ["Q1?", "Q2?"]
    assert prompt_ds["answer"] == ["A1", "A2"]
    # max_evidence_per_prompt clips each prompt's evidence list.
    assert len(evidence_ds) == 4
    assert evidence_ds["sentence"] == ["E1a", "E1b", "E2a", "E2b"]

    # Padded sequences land at max_length; tensors via set_format("torch").
    assert prompt_ds[0]["input_ids"].shape[0] == 16
    assert evidence_ds[0]["input_ids"].shape[0] == 16

    # Prompt tokens (and pad positions) must be masked with -100 in labels.
    prompt_len = len(
        fake_tok("Q1?", truncation=True, max_length=16)["input_ids"]
    )
    label_row = prompt_ds[0]["labels"].tolist()
    assert all(x == -100 for x in label_row[:prompt_len])
    # Pad positions must also be -100.
    assert label_row[-1] == -100

    # Evidence labels: pad positions are -100, real tokens are the input id.
    ev_input = evidence_ds[0]["input_ids"].tolist()
    ev_label = evidence_ds[0]["labels"].tolist()
    real_len = sum(1 for v in evidence_ds[0]["attention_mask"].tolist() if v)
    assert ev_label[:real_len] == ev_input[:real_len]
    assert all(x == -100 for x in ev_label[real_len:])

    # Entailment matrix wires evidence rows back to their prompt rows.
    expected = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.long)
    assert torch.equal(labels, expected)


@pytest.mark.utils
def test_parse_fact_tracing_cfg_samples_when_num_prompts_lt_len(monkeypatch):
    """num_prompts < len(ds) triggers the random.sample branch."""
    rows = [
        {
            "prompt": f"Q{i}?",
            "answer": [f"A{i}"],
            "evidence_sentences": [f"E{i}"],
        }
        for i in range(6)
    ]
    monkeypatch.setattr(
        cp_module, "load_dataset", lambda *a, **kw: _fake_hf_dataset(rows)
    )
    monkeypatch.setattr(
        cp_module, "resolve_tokenizer", lambda cfg: (_FakeTokenizer(), 0)
    )
    cfg = {
        "dataset_str": "fake/ds",
        "tokenizer": {"backend": "hf", "name": "x"},
        "num_prompts": 3,
        "seed": 42,
        "max_length": 8,
        "max_evidence_per_prompt": 5,
    }

    prompt_ds, evidence_ds, labels, _ = (
        FactTracingConfigParser.parse_fact_tracing_cfg(cfg)
    )

    assert len(prompt_ds) == 3
    assert len(evidence_ds) == 3
    # Each evidence belongs to exactly one prompt (one-to-one for these rows).
    assert torch.equal(labels.sum(dim=0), torch.ones(3, dtype=torch.long))
    # Sampling is seed-stable: rerun yields the same prompts.
    prompts2, _, _, _ = FactTracingConfigParser.parse_fact_tracing_cfg(cfg)
    assert prompt_ds["prompt"] == prompts2["prompt"]


@pytest.mark.utils
def test_build_entailment_matrix_basic():
    labels = FactTracingConfigParser._build_entailment_matrix(
        num_queries=3, num_evidence=5, evidence_map=[0, 0, 1, 2, 2]
    )
    expected = torch.tensor(
        [
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1],
        ],
        dtype=torch.long,
    )
    assert torch.equal(labels, expected)


@pytest.mark.utils
def test_build_entailment_matrix_empty():
    labels = FactTracingConfigParser._build_entailment_matrix(
        num_queries=2, num_evidence=0, evidence_map=[]
    )
    assert labels.shape == (2, 0)
    assert labels.dtype == torch.long
