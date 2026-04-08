"""Utils for tokenization of HuggingFace datasets."""

import datasets as hf_datasets  # type: ignore
from transformers import AutoTokenizer


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
