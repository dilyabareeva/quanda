"""Tests for the dattri explainer wrappers.

Two flavours of coverage:

* ``test_dattri_<name>_qnli``: marked ``slow``, runs the real bert-qnli
  fixtures. Uses CUDA when available. Skipped on GitHub Actions.
* ``test_dattri_<name>_simple``: lightweight smoke tests on a tiny dummy
  text classifier, always CPU. Run under GitHub CI.
"""

import os
from typing import Tuple
from unittest.mock import patch

import pytest
import torch
from torch import nn

from quanda.explainers.wrappers import (
    DattriArnoldi,
    DattriEKFAC,
    DattriGradCos,
    DattriGradDot,
    DattriTracInCP,
    DattriTRAK,
)
from tests.models import SimpleTextClassifier


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def vmap_compatible_bert_masking():
    """Disable vmap-incompatible shortcuts in transformers' SDPA masking.

    The HuggingFace BERT forward pass takes a short-circuit path in
    ``transformers.masking_utils`` that calls ``padding_mask.all()`` as a
    Python bool — this is incompatible with ``torch.func.vmap``. Disabling
    the shortcut forces the slower, but vmap-safe, masking path.
    """
    with (
        patch(
            "transformers.masking_utils._ignore_bidirectional_mask_sdpa",
            return_value=False,
        ),
        patch(
            "transformers.masking_utils._ignore_causal_mask_sdpa",
            return_value=False,
        ),
    ):
        yield


BERT_CLASSIFIER_LAYER = [
    "model.classifier.weight",
    "model.classifier.bias",
]
BERT_CLASSIFIER_MODULE = "model.classifier"

SIMPLE_CLASSIFIER_LAYER = [
    "classifier.weight",
    "classifier.bias",
]
SIMPLE_CLASSIFIER_MODULE = "classifier"


# ---------------------------------------------------------------------------
# bert-qnli helpers
# ---------------------------------------------------------------------------


def _qnli_to_tensor_dataset(ds, device: str = "cpu"):
    """Convert a dict-style QNLI HF dataset into an ordered TensorDataset.

    Installed dattri 0.3.0 assumes tuple-style batches in its per-batch
    projector path (``train_batch_data[0].shape[0]``), so we flatten the
    dict into a ``(input_ids, token_type_ids, attention_mask, labels)``
    tuple for dattri compatibility.
    """
    input_ids = torch.tensor([d["input_ids"] for d in ds], device=device)
    token_type_ids = torch.tensor(
        [d["token_type_ids"] for d in ds], device=device
    )
    attention_mask = torch.tensor(
        [d["attention_mask"] for d in ds], device=device
    )
    labels = torch.tensor([d["labels"] for d in ds], device=device)
    return torch.utils.data.TensorDataset(
        input_ids, token_type_ids, attention_mask, labels
    )


def _make_bert_loss_funcs(model):
    """Per-sample / batched loss functions for a bert-qnli model.

    Returns two functions:
      * ``loss_fn_per_sample``: per-sample tuple input (no batch dim). Used
        by TracIn/TRAK-family attributors.
      * ``loss_fn_batched``: batched tuple input (leading batch dim of 1).
        Used by the influence-function family (Arnoldi, EK-FAC).
    """
    ce = nn.CrossEntropyLoss()

    def loss_fn_per_sample(params, batch):
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = torch.func.functional_call(
            model,
            params,
            args=(
                input_ids.unsqueeze(0),
                token_type_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
            ),
        )
        return ce(outputs.logits, labels.unsqueeze(0))

    def loss_fn_batched(params, batch):
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = torch.func.functional_call(
            model,
            params,
            args=(input_ids, token_type_ids, attention_mask),
        )
        return ce(outputs.logits, labels)

    return loss_fn_per_sample, loss_fn_batched


def _bert_correct_probability_fn(model):
    ce = nn.CrossEntropyLoss()

    def m(params, batch):
        input_ids, token_type_ids, attention_mask, labels = batch
        outputs = torch.func.functional_call(
            model,
            params,
            args=(
                input_ids.unsqueeze(0),
                token_type_ids.unsqueeze(0),
                attention_mask.unsqueeze(0),
            ),
        )
        return torch.exp(-ce(outputs.logits, labels.unsqueeze(0)))

    return m


def _qnli_train_and_test_sample(
    train_dataset, test_dataset, device: str
) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    train_tensor_ds = _qnli_to_tensor_dataset(train_dataset, device=device)
    test_sample_ds = _qnli_to_tensor_dataset([test_dataset[0]], device=device)
    return train_tensor_ds, test_sample_ds


# ---------------------------------------------------------------------------
# Simple text-classifier fixtures for CI
# ---------------------------------------------------------------------------


def _simple_dummy_datasets(
    train_size: int = 4, seq_len: int = 8, vocab_size: int = 50
) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    torch.manual_seed(0)
    train = torch.utils.data.TensorDataset(
        torch.randint(1, vocab_size, (train_size, seq_len)),
        torch.ones(train_size, seq_len, dtype=torch.long),
        torch.randint(0, 2, (train_size,)),
    )
    test = torch.utils.data.TensorDataset(
        torch.randint(1, vocab_size, (1, seq_len)),
        torch.ones(1, seq_len, dtype=torch.long),
        torch.randint(0, 2, (1,)),
    )
    return train, test


def _make_simple_loss_funcs(model):
    ce = nn.CrossEntropyLoss()

    def loss_fn_per_sample(params, batch):
        input_ids, attention_mask, labels = batch
        outputs = torch.func.functional_call(
            model,
            params,
            args=(input_ids.unsqueeze(0), attention_mask.unsqueeze(0)),
        )
        return ce(outputs.logits, labels.unsqueeze(0))

    def loss_fn_batched(params, batch):
        input_ids, attention_mask, labels = batch
        outputs = torch.func.functional_call(
            model,
            params,
            args=(input_ids, attention_mask),
        )
        return ce(outputs.logits, labels)

    return loss_fn_per_sample, loss_fn_batched


def _simple_correct_probability_fn(model):
    ce = nn.CrossEntropyLoss()

    def m(params, batch):
        input_ids, attention_mask, labels = batch
        outputs = torch.func.functional_call(
            model,
            params,
            args=(input_ids.unsqueeze(0), attention_mask.unsqueeze(0)),
        )
        return torch.exp(-ce(outputs.logits, labels.unsqueeze(0)))

    return m


# ---------------------------------------------------------------------------
# Slow bert-qnli tests (gated behind `slow` marker; skipped on GitHub CI)
# ---------------------------------------------------------------------------


QNLI_SLOW_MARKS = [
    pytest.mark.slow,
    pytest.mark.explainers,
    pytest.mark.skipif(
        "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
    ),
]


def _apply_marks(marks):
    def decorator(fn):
        for m in reversed(marks):
            fn = m(fn)
        return fn

    return decorator


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_trak_qnli(
    load_qnli_model, load_qnli_dataset, vmap_compatible_bert_masking
):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_tensor_ds, test_sample_ds = _qnli_train_and_test_sample(
        *load_qnli_dataset, device=device
    )

    loss_fn, _ = _make_bert_loss_funcs(model)

    explainer = DattriTRAK(
        model=model,
        train_dataset=train_tensor_ds,
        loss_func=loss_fn,
        correct_probability_func=_bert_correct_probability_fn(model),
        task="text_classification",
        layer_name=BERT_CLASSIFIER_LAYER,
        batch_size=1,
        projector_kwargs={"proj_dim": 16},
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample_ds, targets=None)
    assert explanations.shape == (1, len(train_tensor_ds))


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_tracincp_qnli(
    load_qnli_model, load_qnli_dataset, vmap_compatible_bert_masking
):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_tensor_ds, test_sample_ds = _qnli_train_and_test_sample(
        *load_qnli_dataset, device=device
    )

    loss_fn, _ = _make_bert_loss_funcs(model)

    explainer = DattriTracInCP(
        model=model,
        train_dataset=train_tensor_ds,
        loss_func=loss_fn,
        weight_list=torch.tensor([1.0], device=device),
        task="text_classification",
        layer_name=BERT_CLASSIFIER_LAYER,
        batch_size=1,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample_ds, targets=None)
    assert explanations.shape == (1, len(train_tensor_ds))


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_graddot_qnli(
    load_qnli_model, load_qnli_dataset, vmap_compatible_bert_masking
):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_tensor_ds, test_sample_ds = _qnli_train_and_test_sample(
        *load_qnli_dataset, device=device
    )

    loss_fn, _ = _make_bert_loss_funcs(model)

    explainer = DattriGradDot(
        model=model,
        train_dataset=train_tensor_ds,
        loss_func=loss_fn,
        task="text_classification",
        layer_name=BERT_CLASSIFIER_LAYER,
        batch_size=1,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample_ds, targets=None)
    assert explanations.shape == (1, len(train_tensor_ds))


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_gradcos_qnli(
    load_qnli_model, load_qnli_dataset, vmap_compatible_bert_masking
):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_tensor_ds, test_sample_ds = _qnli_train_and_test_sample(
        *load_qnli_dataset, device=device
    )

    loss_fn, _ = _make_bert_loss_funcs(model)

    explainer = DattriGradCos(
        model=model,
        train_dataset=train_tensor_ds,
        loss_func=loss_fn,
        task="text_classification",
        layer_name=BERT_CLASSIFIER_LAYER,
        batch_size=1,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample_ds, targets=None)
    assert explanations.shape == (1, len(train_tensor_ds))


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_arnoldi_qnli(
    load_qnli_model, load_qnli_dataset, vmap_compatible_bert_masking
):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_tensor_ds, test_sample_ds = _qnli_train_and_test_sample(
        *load_qnli_dataset, device=device
    )

    _, loss_fn = _make_bert_loss_funcs(model)

    explainer = DattriArnoldi(
        model=model,
        train_dataset=train_tensor_ds,
        loss_func=loss_fn,
        task="text_classification",
        layer_name=BERT_CLASSIFIER_LAYER,
        batch_size=1,
        proj_dim=5,
        max_iter=10,
        regularization=1e-3,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample_ds, targets=None)
    assert explanations.shape == (1, len(train_tensor_ds))


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_ekfac_qnli(
    load_qnli_model, load_qnli_dataset, vmap_compatible_bert_masking
):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_tensor_ds, test_sample_ds = _qnli_train_and_test_sample(
        *load_qnli_dataset, device=device
    )

    _, loss_fn = _make_bert_loss_funcs(model)

    explainer = DattriEKFAC(
        model=model,
        train_dataset=train_tensor_ds,
        loss_func=loss_fn,
        task="text_classification",
        module_name=BERT_CLASSIFIER_MODULE,
        batch_size=1,
        damping=1e-2,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample_ds, targets=None)
    assert explanations.shape == (1, len(train_tensor_ds))


# ---------------------------------------------------------------------------
# Fast CI smoke tests on a tiny text classifier (always CPU)
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_setup():
    model = SimpleTextClassifier(vocab_size=50, hidden_size=16)
    model.eval()
    train_ds, test_ds = _simple_dummy_datasets()
    return model, train_ds, test_ds


@pytest.mark.explainers
def test_dattri_trak_simple(simple_setup):
    model, train_ds, test_ds = simple_setup
    loss_fn, _ = _make_simple_loss_funcs(model)

    explainer = DattriTRAK(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        correct_probability_func=_simple_correct_probability_fn(model),
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        projector_kwargs={"proj_dim": 8},
        device="cpu",
    )
    explanations = explainer.explain(test_data=test_ds, targets=None)
    assert explanations.shape == (1, len(train_ds))


@pytest.mark.explainers
def test_dattri_tracincp_simple(simple_setup):
    model, train_ds, test_ds = simple_setup
    loss_fn, _ = _make_simple_loss_funcs(model)

    explainer = DattriTracInCP(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        weight_list=torch.tensor([1.0]),
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        device="cpu",
    )
    explanations = explainer.explain(test_data=test_ds, targets=None)
    assert explanations.shape == (1, len(train_ds))


@pytest.mark.explainers
def test_dattri_graddot_simple(simple_setup):
    model, train_ds, test_ds = simple_setup
    loss_fn, _ = _make_simple_loss_funcs(model)

    explainer = DattriGradDot(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        device="cpu",
    )
    explanations = explainer.explain(test_data=test_ds, targets=None)
    assert explanations.shape == (1, len(train_ds))


@pytest.mark.explainers
def test_dattri_gradcos_simple(simple_setup):
    model, train_ds, test_ds = simple_setup
    loss_fn, _ = _make_simple_loss_funcs(model)

    explainer = DattriGradCos(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        device="cpu",
    )
    explanations = explainer.explain(test_data=test_ds, targets=None)
    assert explanations.shape == (1, len(train_ds))


@pytest.mark.explainers
def test_dattri_arnoldi_simple(simple_setup):
    model, train_ds, test_ds = simple_setup
    _, loss_fn = _make_simple_loss_funcs(model)

    explainer = DattriArnoldi(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        proj_dim=5,
        max_iter=10,
        regularization=1e-3,
        device="cpu",
    )
    explanations = explainer.explain(test_data=test_ds, targets=None)
    assert explanations.shape == (1, len(train_ds))


@pytest.mark.explainers
def test_dattri_ekfac_simple(simple_setup):
    model, train_ds, test_ds = simple_setup
    _, loss_fn = _make_simple_loss_funcs(model)

    explainer = DattriEKFAC(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        task="text_classification",
        module_name=SIMPLE_CLASSIFIER_MODULE,
        batch_size=1,
        damping=1e-2,
        device="cpu",
    )
    explanations = explainer.explain(test_data=test_ds, targets=None)
    assert explanations.shape == (1, len(train_ds))
