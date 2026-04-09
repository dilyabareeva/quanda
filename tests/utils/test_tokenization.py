"""Tests for quanda.utils.tokenization."""

import datasets as hf_datasets
import pytest
import torch

from quanda.utils.tokenization import tokenize_dataset


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
