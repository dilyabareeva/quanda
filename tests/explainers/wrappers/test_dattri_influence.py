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
from quanda.explainers.wrappers.dattri_influence import (
    DattriInfluence,
    _resolve_device,
    _wrap_checkpoints_load_func,
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


def _simple_per_sample_loss_builder(model):
    ce = nn.CrossEntropyLoss()

    def loss_fn(params, batch):
        input_ids, labels = batch
        outputs = torch.func.functional_call(
            model, params, args=(input_ids.unsqueeze(0),)
        )
        return ce(outputs.logits, labels.unsqueeze(0))

    return loss_fn


def _simple_batched_loss_builder(model):
    ce = nn.CrossEntropyLoss()

    def loss_fn(params, batch):
        input_ids, labels = batch
        outputs = torch.func.functional_call(model, params, args=(input_ids,))
        return ce(outputs.logits, labels)

    return loss_fn


def _simple_correct_probability_builder(model):
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

    explainer = DattriTRAK(
        model=model,
        train_dataset=train_ds,
        loss_func=bert_classification_per_sample_loss,
        correct_probability_func=bert_classification_correct_probability,
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

    explainer = DattriTracInCP(
        model=model,
        train_dataset=train_ds,
        loss_func=bert_classification_per_sample_loss,
        learning_rate=1.0,
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

    explainer = DattriGradDot(
        model=model,
        train_dataset=train_ds,
        loss_func=bert_classification_per_sample_loss,
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

    explainer = DattriGradCos(
        model=model,
        train_dataset=train_ds,
        loss_func=bert_classification_per_sample_loss,
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

    explainer = DattriArnoldi(
        model=model,
        train_dataset=train_ds,
        loss_func=bert_classification_batched_loss,
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

    explainer = DattriEKFAC(
        model=model,
        train_dataset=train_ds,
        loss_func=bert_classification_batched_loss,
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

    explainer = DattriTRAK(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_per_sample_loss_builder,
        correct_probability_func=_simple_correct_probability_builder,
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

    explainer = DattriTracInCP(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_per_sample_loss_builder,
        learning_rate=1.0,
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

    explainer = DattriGradDot(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_per_sample_loss_builder,
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

    explainer = DattriGradCos(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_per_sample_loss_builder,
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

    explainer = DattriArnoldi(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_batched_loss_builder,
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

    explainer = DattriEKFAC(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_batched_loss_builder,
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


@pytest.mark.explainers
def test_dattri_tracincp_self_influence(simple_setup):
    """Base-class ``self_influence`` delegates to ``attributor.self_attribute``."""
    model, train_ds, _ = simple_setup
    explainer = DattriTracInCP(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_per_sample_loss_builder,
        learning_rate=1.0,
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        device="cpu",
    )
    scores = explainer.self_influence(batch_size=1)
    assert scores.shape[0] == len(train_ds)


@pytest.mark.explainers
def test_dattri_trak_without_cache_paths(simple_setup):
    """``use_cache=False`` falls back through the ``(test, train)`` call
    and the manual self-attribute loop."""
    model, train_ds, test_ds = simple_setup
    explainer = DattriTRAK(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_per_sample_loss_builder,
        correct_probability_func=_simple_correct_probability_builder,
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        projector_kwargs={"proj_dim": 8},
        use_cache=False,
        device="cpu",
    )
    assert explainer.attributor.full_train_dataloader is None

    test_input_ids, test_labels = test_ds[0]
    explanations = explainer.explain(
        test_data=test_input_ids.unsqueeze(0),
        targets=test_labels.unsqueeze(0),
    )
    assert explanations.shape == (1, len(train_ds))

    scores = explainer.self_influence(batch_size=1)
    assert scores.shape[0] == len(train_ds)


@pytest.mark.explainers
@pytest.mark.parametrize(
    "cls",
    [DattriGradDot, DattriGradCos, DattriArnoldi],
)
def test_dattri_wrappers_keep_last_checkpoint_when_multiple(cls, simple_setup):
    """Passing >1 checkpoints must collapse to the final one."""
    model, train_ds, test_ds = simple_setup

    ckpt_paths = ["ckpt_a.pt", "ckpt_b.pt"]
    loader_calls = []

    def loader(m, path):
        loader_calls.append(path)

    kwargs = dict(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_batched_loss_builder
        if cls is DattriArnoldi
        else _simple_per_sample_loss_builder,
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        checkpoints=ckpt_paths,
        checkpoints_load_func=loader,
        device="cpu",
    )
    if cls is DattriArnoldi:
        kwargs.update(proj_dim=5, max_iter=2, regularization=1e-3)

    explainer = cls(**kwargs)
    assert explainer.checkpoints == [ckpt_paths[-1]]
    assert all(p == ckpt_paths[-1] for p in loader_calls)


@pytest.mark.explainers
def test_resolve_device_passthrough_and_param_fallback():
    model = nn.Linear(2, 2)
    assert _resolve_device(model, "cuda") == "cuda"
    # fall back to the param's device
    assert _resolve_device(model, None) == str(next(model.parameters()).device)


@pytest.mark.explainers
def test_resolve_device_defaults_to_cpu_when_model_has_no_params():
    class _NoParams(nn.Module):
        def forward(self, x):
            return x

    assert _resolve_device(_NoParams(), None) == "cpu"


@pytest.mark.explainers
def test_wrap_checkpoints_load_func_invokes_inner_and_returns_model():
    seen = []

    def inner(model, ckpt):
        seen.append((model, ckpt))

    wrapped = _wrap_checkpoints_load_func(inner)
    model = nn.Linear(1, 1)
    returned = wrapped(model, "some-ckpt")
    assert returned is model
    assert seen == [(model, "some-ckpt")]


@pytest.mark.explainers
def test_create_test_dataset_tensor_without_targets_raises(simple_setup):
    model, train_ds, _ = simple_setup
    explainer = DattriTracInCP(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_per_sample_loss_builder,
        learning_rate=1.0,
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        device="cpu",
    )
    with pytest.raises(ValueError, match="Targets required"):
        explainer._create_test_dataset(torch.zeros(1, 8), targets=None)


@pytest.mark.explainers
def test_create_test_dataset_tensor_accepts_list_targets(simple_setup):
    model, train_ds, _ = simple_setup
    explainer = DattriTracInCP(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_per_sample_loss_builder,
        learning_rate=1.0,
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        device="cpu",
    )
    ds = explainer._create_test_dataset(
        torch.zeros(2, 8, dtype=torch.long), targets=[0, 1]
    )
    assert isinstance(ds, torch.utils.data.TensorDataset)
    assert torch.equal(ds.tensors[1], torch.tensor([0, 1]))


@pytest.mark.explainers
def test_create_test_dataset_dict_builds_hf_dataset(simple_setup):
    import datasets as hf_datasets

    model, train_ds, _ = simple_setup
    explainer = DattriTracInCP(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_per_sample_loss_builder,
        learning_rate=1.0,
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        device="cpu",
    )
    ds = explainer._create_test_dataset(
        {"input_ids": torch.tensor([[1, 2, 3]])},
        targets=torch.tensor([1]),
    )
    assert isinstance(ds, hf_datasets.Dataset)
    assert "labels" in ds.features
    assert ds["labels"] == [1]


@pytest.mark.explainers
def test_create_test_dataset_unsupported_type_raises(simple_setup):
    model, train_ds, _ = simple_setup
    explainer = DattriTracInCP(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_per_sample_loss_builder,
        learning_rate=1.0,
        task="text_classification",
        layer_name=SIMPLE_CLASSIFIER_LAYER,
        batch_size=1,
        device="cpu",
    )
    with pytest.raises(ValueError, match="Unsupported test_data type"):
        explainer._create_test_dataset("not a tensor", targets=None)


@pytest.mark.explainers
def test_dattri_influence_is_abstract_explainer_subclass():
    """Sanity check so the import path exercised by the helpers is live."""
    assert issubclass(DattriTracInCP, DattriInfluence)


@pytest.mark.explainers
def test_dattri_ekfac_keeps_last_checkpoint_when_multiple(simple_setup):
    model, train_ds, _ = simple_setup
    ckpt_paths = ["a", "b"]

    def loader(m, path):
        pass

    explainer = DattriEKFAC(
        model=model,
        train_dataset=train_ds,
        loss_func=_simple_batched_loss_builder,
        task="text_classification",
        module_name=SIMPLE_CLASSIFIER_MODULE,
        batch_size=1,
        damping=1e-2,
        checkpoints=ckpt_paths,
        checkpoints_load_func=loader,
        device="cpu",
    )
    assert explainer.checkpoints == [ckpt_paths[-1]]
