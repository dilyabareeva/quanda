"""Tests for quanda.utils.tokenization."""

import datasets as hf_datasets
import pytest
import torch

from quanda.utils.tokenization import tokenize_dataset, _TikTokenHFAdapter, resolve_tokenizer

@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, ds_key, ds_kwargs, cfg, checks",
    [
        (
            "basic_two_fields",
            "two_field_dataset",
            {"n": 8},
            {
                "name": "bert-base-cased",
                "text_fields": ["question", "sentence"],
            },
            {
                "len": 8,
                "has_cols": ["input_ids", "attention_mask", "labels"],
                "missing_cols": [],
            },
        ),
        (
            "custom_max_length",
            "two_field_dataset",
            {"n": 4},
            {
                "name": "bert-base-cased",
                "text_fields": ["question", "sentence"],
                "max_length": 32,
            },
            {
                "len": 4,
                "has_cols": ["input_ids"],
                "missing_cols": [],
                "seq_len": 32,
            },
        ),
        (
            "single_field",
            "single_field_dataset",
            {},
            {
                "name": "bert-base-cased",
                "text_fields": ["text"],
                "max_length": 16,
            },
            {
                "len": 2,
                "has_cols": ["input_ids"],
                "missing_cols": [],
            },
        ),
        (
            "custom_label_field",
            "custom_label_dataset",
            {},
            {
                "name": "bert-base-cased",
                "text_fields": ["text"],
                "label_field": "my_label",
                "max_length": 16,
            },
            {
                "len": 3,
                "has_cols": ["labels"],
                "missing_cols": [],
                "first_label": 0,
                "last_label": 2,
            },
        ),
        (
            "removes_original_cols",
            "two_field_dataset",
            {"n": 4},
            {
                "name": "bert-base-cased",
                "text_fields": ["question", "sentence"],
                "max_length": 16,
            },
            {
                "len": 4,
                "has_cols": ["input_ids"],
                "missing_cols": ["question", "sentence"],
            },
        ),
    ],
)
def test_tokenize_dataset(test_id, ds_key, ds_kwargs, cfg, checks, request):
    ds_factory = request.getfixturevalue(ds_key)
    result = tokenize_dataset(ds_factory(**ds_kwargs), cfg)

    assert isinstance(result, hf_datasets.Dataset)
    assert len(result) == checks["len"]

    for col in checks["has_cols"]:
        assert col in result.column_names

    for col in checks.get("missing_cols", []):
        assert col not in result.column_names

    if "seq_len" in checks:
        assert result[0]["input_ids"].shape[0] == checks["seq_len"]

    if "first_label" in checks:
        assert result[0]["labels"].item() == checks["first_label"]
    if "last_label" in checks:
        assert result[-1]["labels"].item() == checks["last_label"]

    # All results should be torch tensors
    assert isinstance(result[0]["input_ids"], torch.Tensor)


@pytest.fixture
def adapter():
    return _TikTokenHFAdapter(encoding_name="gpt2")


@pytest.mark.utils
def test_init_sets_pad_token_to_eot(adapter):
    """pad_token_id mirrors the tiktoken encoding's eot token."""
    import tiktoken

    assert adapter.pad_token_id == tiktoken.get_encoding("gpt2").eot_token


@pytest.mark.utils
def test_call_no_padding_no_truncation(adapter):
    out = adapter("hello world")
    assert isinstance(out["input_ids"], list)
    assert len(out["input_ids"]) > 0
    assert out["attention_mask"] == [1] * len(out["input_ids"])
    assert adapter.pad_token_id not in out["input_ids"]


@pytest.mark.utils
def test_call_truncates_when_over_max_length(adapter):
    long_text = "tokenization " * 50
    full = adapter(long_text)["input_ids"]
    truncated = adapter(long_text, truncation=True, max_length=8)
    assert len(truncated["input_ids"]) == 8
    assert truncated["input_ids"] == full[:8]
    assert truncated["attention_mask"] == [1] * 8


@pytest.mark.utils
def test_call_pads_to_max_length(adapter):
    out = adapter("hi", padding="max_length", truncation=True, max_length=16)
    real_len = sum(out["attention_mask"])
    assert len(out["input_ids"]) == 16
    assert len(out["attention_mask"]) == 16
    assert out["input_ids"][real_len:] == [adapter.pad_token_id] * (
        16 - real_len
    )
    assert out["attention_mask"] == [1] * real_len + [0] * (16 - real_len)


@pytest.mark.utils
def test_call_no_padding_when_padding_false(adapter):
    """padding=False keeps the variable-length output."""
    out = adapter("hi there", padding=False, max_length=64)
    assert len(out["input_ids"]) < 64
    assert all(m == 1 for m in out["attention_mask"])


@pytest.mark.utils
def test_call_ignores_unknown_kwargs(adapter):
    """The HF tokenizer call site passes extra kwargs we should swallow."""
    out = adapter("x", return_tensors="pt", add_special_tokens=False)
    assert "input_ids" in out
    assert "attention_mask" in out


@pytest.mark.utils
def test_resolve_tokenizer_tiktoken_backend():
    """resolve_tokenizer routes the tiktoken backend through the adapter."""
    tok, pad_id = resolve_tokenizer(
        {"backend": "tiktoken", "encoding": "gpt2"}
    )
    assert isinstance(tok, _TikTokenHFAdapter)
    assert pad_id == tok.pad_token_id


@pytest.mark.utils
def test_resolve_tokenizer_tiktoken_default_encoding():
    """Missing ``encoding`` key falls back to the gpt2 encoding."""
    tok, _ = resolve_tokenizer({"backend": "tiktoken"})
    assert isinstance(tok, _TikTokenHFAdapter)


@pytest.mark.utils
def test_resolve_tokenizer_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown tokenizer backend"):
        resolve_tokenizer({"backend": "nope"})
