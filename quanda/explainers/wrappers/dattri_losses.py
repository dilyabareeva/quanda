"""Loss/probability builders for dattri explainer wrappers."""

from typing import Callable

import torch
from torch import nn


def _bert_4d_attention_mask(
    mask_2d: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Build the 4D additive mask HF BERT's eager attention expects.

    A 2D ``(batch, seq_len)`` mask would route through
    ``transformers.masking_utils.create_bidirectional_mask``, whose skip
    path calls ``padding_mask.all()`` as a Python bool — incompatible
    with ``torch.func.vmap``. Passing a 4D tensor hits the early-exit in
    ``_preprocess_mask_arguments`` and is used as-is.
    """
    min_val = torch.finfo(dtype).min
    return (1.0 - mask_2d.to(dtype))[:, None, None, :] * min_val


def _bert_forward_kwargs(
    input_ids: torch.Tensor,
    token_type_ids: torch.Tensor,
    attention_mask_4d: torch.Tensor,
) -> dict:
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask_4d,
        "token_type_ids": token_type_ids,
    }


def bert_classification_per_sample_loss(
    model: nn.Module,
) -> Callable:
    """Build a per-sample CE loss for a HF BERT classification model.

    Expects batches of ``(input_ids, token_type_ids, attention_mask,
    labels)``, each without a leading batch dim. Suitable for TracIn-
    and TRAK-family attributors.

    Uses keyword args for ``torch.func.functional_call`` so the builder
    is agnostic to the model's positional-argument order.
    """
    ce = nn.CrossEntropyLoss()
    dtype = next(model.parameters()).dtype

    def loss_fn(params, batch):
        input_ids, token_type_ids, attention_mask, labels = batch
        am_4d = _bert_4d_attention_mask(attention_mask.unsqueeze(0), dtype)
        outputs = torch.func.functional_call(
            model,
            params,
            args=(),
            kwargs=_bert_forward_kwargs(
                input_ids.unsqueeze(0),
                token_type_ids.unsqueeze(0),
                am_4d,
            ),
        )
        logits = getattr(outputs, "logits", outputs)
        return ce(logits, labels.unsqueeze(0))

    return loss_fn


def bert_classification_batched_loss(
    model: nn.Module,
) -> Callable:
    """Build a batched CE loss for a HF BERT classification model.

    Expects batches of ``(input_ids, token_type_ids, attention_mask,
    labels)`` each carrying a leading batch dim. Suitable for the
    influence-function family (Arnoldi, EK-FAC).
    """
    ce = nn.CrossEntropyLoss()
    dtype = next(model.parameters()).dtype

    def loss_fn(params, batch):
        input_ids, token_type_ids, attention_mask, labels = batch
        am_4d = _bert_4d_attention_mask(attention_mask, dtype)
        outputs = torch.func.functional_call(
            model,
            params,
            args=(),
            kwargs=_bert_forward_kwargs(input_ids, token_type_ids, am_4d),
        )
        logits = getattr(outputs, "logits", outputs)
        return ce(logits, labels)

    return loss_fn


def bert_classification_correct_probability(
    model: nn.Module,
) -> Callable:
    """Build the per-sample correct-class probability for HF BERT.

    Used by dattri's TRAK attributor. Expects the same per-sample batch
    layout as :func:`bert_classification_per_sample_loss`.
    """
    ce = nn.CrossEntropyLoss()
    dtype = next(model.parameters()).dtype

    def prob_fn(params, batch):
        input_ids, token_type_ids, attention_mask, labels = batch
        am_4d = _bert_4d_attention_mask(attention_mask.unsqueeze(0), dtype)
        outputs = torch.func.functional_call(
            model,
            params,
            args=(),
            kwargs=_bert_forward_kwargs(
                input_ids.unsqueeze(0),
                token_type_ids.unsqueeze(0),
                am_4d,
            ),
        )
        logits = getattr(outputs, "logits", outputs)
        return torch.exp(-ce(logits, labels.unsqueeze(0)))

    return prob_fn


__all__ = [
    "bert_classification_per_sample_loss",
    "bert_classification_batched_loss",
    "bert_classification_correct_probability",
]
