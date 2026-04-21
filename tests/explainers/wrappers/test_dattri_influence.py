"""Tests for the dattri explainer wrappers.

* ``test_dattri_<name>_qnli``: marked ``slow``, runs the real bert-qnli
  fixtures. Uses CUDA when available. Skipped on GitHub Actions.
* ``test_dattri_<name>_simple``: lightweight smoke tests on a tiny dummy
  text classifier, always CPU. Run under GitHub CI.
"""

import os
from typing import Tuple

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
    bert_classification_batched_loss,
    bert_classification_correct_probability,
    bert_classification_per_sample_loss,
)
from quanda.utils.datasets.dataset_handlers import (
    HuggingFaceSequenceDatasetHandler,
)
from tests.models import SimpleTextClassifier


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


BERT_CLASSIFIER_LAYER = [
    "model.classifier.weight",
    "model.classifier.bias",
]
BERT_CLASSIFIER_MODULE = "model.classifier"

BERT_ARNOLDI_LAYERS = [
    "model.bert.pooler.dense.weight",
    "model.bert.pooler.dense.bias",
    "model.classifier.weight",
    "model.classifier.bias",
]

SIMPLE_CLASSIFIER_LAYER = [
    "classifier.weight",
    "classifier.bias",
]
SIMPLE_CLASSIFIER_MODULE = "classifier"


# ---------------------------------------------------------------------------
# bert-qnli helpers
# ---------------------------------------------------------------------------

BERT_SEQ_HANDLER = HuggingFaceSequenceDatasetHandler(
    input_keys=("input_ids", "token_type_ids", "attention_mask"),
)


def _simple_dummy_datasets(
    train_size: int = 4, seq_len: int = 8, vocab_size: int = 50
) -> Tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    torch.manual_seed(0)
    train = torch.utils.data.TensorDataset(
        torch.randint(1, vocab_size, (train_size, seq_len)),
        torch.randint(0, 2, (train_size,)),
    )
    test = torch.utils.data.TensorDataset(
        torch.randint(1, vocab_size, (1, seq_len)),
        torch.randint(0, 2, (1,)),
    )
    return train, test


def _make_simple_loss_funcs(model):
    ce = nn.CrossEntropyLoss()

    def loss_fn_per_sample(params, batch):
        input_ids, labels = batch
        outputs = torch.func.functional_call(
            model, params, args=(input_ids.unsqueeze(0),)
        )
        return ce(outputs.logits, labels.unsqueeze(0))

    def loss_fn_batched(params, batch):
        input_ids, labels = batch
        outputs = torch.func.functional_call(model, params, args=(input_ids,))
        return ce(outputs.logits, labels)

    return loss_fn_per_sample, loss_fn_batched


def _simple_correct_probability_fn(model):
    ce = nn.CrossEntropyLoss()

    def m(params, batch):
        input_ids, labels = batch
        outputs = torch.func.functional_call(
            model, params, args=(input_ids.unsqueeze(0),)
        )
        return torch.exp(-ce(outputs.logits, labels.unsqueeze(0)))

    return m


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
def test_dattri_trak_qnli(load_qnli_model, load_qnli_dataset):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_ds, test_ds = load_qnli_dataset
    test_sample = test_ds[:1]

    loss_fn = bert_classification_per_sample_loss(model)

    explainer = DattriTRAK(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        correct_probability_func=bert_classification_correct_probability(
            model
        ),
        task="text_classification",
        layer_name=BERT_CLASSIFIER_LAYER,
        batch_size=1,
        projector_kwargs={"proj_dim": 16},
        collate_fn=BERT_SEQ_HANDLER.collate,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample, targets=None)
    assert explanations.shape == (1, len(train_ds))


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_tracincp_qnli(load_qnli_model, load_qnli_dataset):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_ds, test_ds = load_qnli_dataset
    test_sample = test_ds[:1]

    loss_fn = bert_classification_per_sample_loss(model)

    explainer = DattriTracInCP(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        weight_list=torch.tensor([1.0], device=device),
        task="text_classification",
        layer_name=BERT_CLASSIFIER_LAYER,
        batch_size=1,
        collate_fn=BERT_SEQ_HANDLER.collate,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample, targets=None)
    assert explanations.shape == (1, len(train_ds))


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_graddot_qnli(load_qnli_model, load_qnli_dataset):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_ds, test_ds = load_qnli_dataset
    test_sample = test_ds[:1]

    loss_fn = bert_classification_per_sample_loss(model)

    explainer = DattriGradDot(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        task="text_classification",
        layer_name=BERT_CLASSIFIER_LAYER,
        batch_size=1,
        collate_fn=BERT_SEQ_HANDLER.collate,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample, targets=None)
    assert explanations.shape == (1, len(train_ds))


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_gradcos_qnli(load_qnli_model, load_qnli_dataset):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_ds, test_ds = load_qnli_dataset
    test_sample = test_ds[:1]

    loss_fn = bert_classification_per_sample_loss(model)

    explainer = DattriGradCos(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        task="text_classification",
        layer_name=BERT_CLASSIFIER_LAYER,
        batch_size=1,
        collate_fn=BERT_SEQ_HANDLER.collate,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample, targets=None)
    assert explanations.shape == (1, len(train_ds))


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_arnoldi_qnli(load_qnli_model, load_qnli_dataset):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_ds, test_ds = load_qnli_dataset
    test_sample = test_ds[:1]

    loss_fn = bert_classification_batched_loss(model)

    explainer = DattriArnoldi(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        task="text_classification",
        layer_name=BERT_ARNOLDI_LAYERS,
        batch_size=1,
        proj_dim=10,
        max_iter=3,
        regularization=1e-3,
        collate_fn=BERT_SEQ_HANDLER.collate,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample, targets=None)
    assert explanations.shape == (1, len(train_ds))


@_apply_marks(QNLI_SLOW_MARKS)
def test_dattri_ekfac_qnli(load_qnli_model, load_qnli_dataset):
    device = _device()
    model = load_qnli_model.to(device)
    model.eval()

    train_ds, test_ds = load_qnli_dataset
    test_sample = test_ds[:1]

    loss_fn = bert_classification_batched_loss(model)

    explainer = DattriEKFAC(
        model=model,
        train_dataset=train_ds,
        loss_func=loss_fn,
        task="text_classification",
        module_name=BERT_CLASSIFIER_MODULE,
        batch_size=1,
        damping=1e-2,
        max_iter=1,
        collate_fn=BERT_SEQ_HANDLER.collate,
        device=device,
    )

    explanations = explainer.explain(test_data=test_sample, targets=None)
    assert explanations.shape == (1, len(train_ds))


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
    test_input_ids, test_labels = test_ds[0]
    explanations = explainer.explain(
        test_data=test_input_ids.unsqueeze(0),
        targets=test_labels.unsqueeze(0),
    )
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
    test_input_ids, test_labels = test_ds[0]
    explanations = explainer.explain(
        test_data=test_input_ids.unsqueeze(0),
        targets=test_labels.unsqueeze(0),
    )
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
    test_input_ids, test_labels = test_ds[0]
    explanations = explainer.explain(
        test_data=test_input_ids.unsqueeze(0),
        targets=test_labels.unsqueeze(0),
    )
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
    test_input_ids, test_labels = test_ds[0]
    explanations = explainer.explain(
        test_data=test_input_ids.unsqueeze(0),
        targets=test_labels.unsqueeze(0),
    )
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
    test_input_ids, test_labels = test_ds[0]
    explanations = explainer.explain(
        test_data=test_input_ids.unsqueeze(0),
        targets=test_labels.unsqueeze(0),
    )
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
    test_input_ids, test_labels = test_ds[0]
    explanations = explainer.explain(
        test_data=test_input_ids.unsqueeze(0),
        targets=test_labels.unsqueeze(0),
    )
    assert explanations.shape == (1, len(train_ds))
