"""Utils for tokenization of HuggingFace datasets."""

from typing import Any, Tuple

import datasets as hf_datasets  # type: ignore
from transformers import AutoTokenizer


class _TikTokenHFAdapter:
    """HF-tokenizer-shaped adapter around a tiktoken encoding.

    Exposes the subset of ``AutoTokenizer.__call__`` that
    :class:`~quanda.benchmarks.config_parser.FactTracingConfigParser`
    relies on — ``(text, padding, truncation, max_length)`` returning
    ``{"input_ids": ..., "attention_mask": ...}`` — so the parser can
    treat tiktoken and HF tokenizers uniformly.
    """

    def __init__(self, encoding_name: str = "gpt2"):
        import tiktoken  # type: ignore

        self._enc = tiktoken.get_encoding(encoding_name)
        self.pad_token_id: int = self._enc.eot_token

    def __call__(
        self,
        text: str,
        padding: Any = False,
        truncation: bool = False,
        max_length: int = None,
        **_: Any,
    ) -> dict:
        ids = self._enc.encode_ordinary(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if padding == "max_length" and max_length is not None:
            pad_len = max_length - len(ids)
            input_ids = ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * len(ids) + [0] * pad_len
        else:
            input_ids = list(ids)
            attention_mask = [1] * len(ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def _hf_tokenizer(tokenizer_name: str):
    """Return a HF AutoTokenizer, ensuring a pad token is set."""
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def resolve_tokenizer(tokenizer_cfg: dict) -> Tuple[Any, int]:
    """Resolve a tokenizer config to ``(tokenizer, pad_token_id)``.

    ``tokenizer`` exposes HF's
    ``__call__(text, padding, truncation, max_length) -> {"input_ids",
    "attention_mask"}``. Supported backends:

    - ``backend: hf`` with ``name`` (HF tokenizer repo) — returns the
      ``AutoTokenizer`` directly.
    - ``backend: tiktoken`` with ``encoding`` (default ``gpt2``) —
      returns a :class:`_TikTokenHFAdapter` with the same interface.
    """
    backend = tokenizer_cfg.get("backend", "hf")
    if backend == "tiktoken":
        adapter = _TikTokenHFAdapter(tokenizer_cfg.get("encoding", "gpt2"))
        return adapter, adapter.pad_token_id
    if backend == "hf":
        tok = _hf_tokenizer(tokenizer_cfg["name"])
        return tok, tok.pad_token_id
    raise ValueError(f"Unknown tokenizer backend: {backend}")


def tokenize_dataset(
    hf_dataset: hf_datasets.Dataset,
    tokenizer_cfg: dict,
) -> hf_datasets.Dataset:
    """Tokenize an HF dataset for transformer models.

    Parameters
    ----------
    hf_dataset : datasets.Dataset
        Raw HuggingFace dataset.
    tokenizer_cfg : dict
        Keys: ``name``, ``text_fields``, ``max_length``,
        ``label_field``.

    Returns
    -------
    datasets.Dataset
        Tokenized dataset formatted as torch tensors.

    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_cfg["name"])
    text_fields = tokenizer_cfg["text_fields"]
    max_length = tokenizer_cfg.get("max_length", 128)
    label_field = tokenizer_cfg.get("label_field", "label")

    def tokenize_fn(examples: dict) -> dict:
        texts = [examples[f] for f in text_fields]
        result = tokenizer(
            *texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        result["labels"] = examples[label_field]
        return result

    tokenized = hf_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=hf_dataset.column_names,
    )
    tokenized.set_format("torch")
    return tokenized
