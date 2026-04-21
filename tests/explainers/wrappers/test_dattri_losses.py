"""Unit tests for the BERT loss builders in
``quanda.explainers.wrappers.dattri_losses``.

The builders return closures that call ``torch.func.functional_call`` with
a 4D attention mask. We exercise those closures directly with a tiny
BERT-shaped stub model so coverage does not depend on loading the real
``gchhablani/bert-base-cased-finetuned-qnli`` checkpoint.
"""

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import (  # type: ignore
    SequenceClassifierOutput,
)

from quanda.explainers.wrappers.dattri_losses import (
    _bert_4d_attention_mask,
    _bert_forward_kwargs,
    bert_classification_batched_loss,
    bert_classification_correct_probability,
    bert_classification_per_sample_loss,
)


class _BertLikeStub(nn.Module):
    """Tiny stand-in for HF BERT's eager-attention forward signature."""

    def __init__(self, vocab_size=20, hidden=4, num_labels=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.type_emb = nn.Embedding(2, hidden)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # Use the 4D mask so ``_bert_4d_attention_mask`` execution is
        # observed end-to-end.
        assert attention_mask.dim() == 4
        x = self.embedding(input_ids) + self.type_emb(token_type_ids)
        logits = self.classifier(x.mean(dim=1))
        # ``logits`` attribute mirrors HF outputs.
        return SequenceClassifierOutput(logits=logits)


def _params(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _sample_per_sample():
    input_ids = torch.tensor([3, 7, 1, 0])
    token_type_ids = torch.zeros(4, dtype=torch.long)
    attention_mask = torch.tensor([1, 1, 1, 0])
    labels = torch.tensor(1)
    return input_ids, token_type_ids, attention_mask, labels


def _sample_batched(batch_size=2):
    input_ids = torch.tensor([[3, 7, 1, 0], [2, 2, 1, 0]])[:batch_size]
    token_type_ids = torch.zeros(batch_size, 4, dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])[:batch_size]
    labels = torch.tensor([1, 0])[:batch_size]
    return input_ids, token_type_ids, attention_mask, labels


@pytest.mark.explainers
def test_bert_4d_attention_mask_masks_padding_positions():
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]])
    am_4d = _bert_4d_attention_mask(mask, torch.float32)

    assert am_4d.shape == (2, 1, 1, 4)
    min_val = torch.finfo(torch.float32).min
    assert am_4d[0, 0, 0, 0].item() == 0.0
    assert am_4d[0, 0, 0, 2].item() == pytest.approx(min_val)
    assert am_4d[1, 0, 0, 3].item() == pytest.approx(min_val)


@pytest.mark.explainers
def test_bert_forward_kwargs_packs_inputs():
    ids = torch.zeros(1, 2, dtype=torch.long)
    tti = torch.zeros(1, 2, dtype=torch.long)
    am = torch.zeros(1, 1, 1, 2)
    kw = _bert_forward_kwargs(ids, tti, am)
    assert set(kw) == {"input_ids", "attention_mask", "token_type_ids"}
    assert kw["attention_mask"] is am


@pytest.mark.explainers
def test_per_sample_loss_returns_finite_scalar():
    torch.manual_seed(0)
    model = _BertLikeStub()
    loss_fn = bert_classification_per_sample_loss(model)

    loss = loss_fn(_params(model), _sample_per_sample())

    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.explainers
def test_batched_loss_matches_manual_cross_entropy():
    torch.manual_seed(0)
    model = _BertLikeStub()
    loss_fn = bert_classification_batched_loss(model)

    batch = _sample_batched()
    loss = loss_fn(_params(model), batch)

    # Recompute manually to assert the closure wired up the model correctly.
    input_ids, token_type_ids, attention_mask, labels = batch
    am_4d = _bert_4d_attention_mask(attention_mask, torch.float32)
    manual_logits = model(
        input_ids=input_ids,
        attention_mask=am_4d,
        token_type_ids=token_type_ids,
    ).logits
    manual_loss = nn.CrossEntropyLoss()(manual_logits, labels)
    assert torch.allclose(loss, manual_loss)


@pytest.mark.explainers
def test_correct_probability_is_in_unit_interval():
    torch.manual_seed(0)
    model = _BertLikeStub()
    prob_fn = bert_classification_correct_probability(model)

    p = prob_fn(_params(model), _sample_per_sample())

    assert p.ndim == 0
    assert 0.0 <= p.item() <= 1.0
