"""Text/NLP fixtures: GLUE/QNLI, GPT-2, Kronfluence LM tasks, fact tracing."""

from itertools import chain
from typing import Dict, List

import datasets
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from kronfluence.task import Task  # type: ignore
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from quanda.benchmarks.config_parser import FactTracingConfigParser
from tests.models import (
    SequenceClassificationModel,
    SimpleCausalLM,
    SimpleTextClassifier,
    TinyGPT2,
)

# Copied from huggingface/transformers run_glue.py.
GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

QNLI_TRAIN_SET_SIZE = 4
QNLI_VAL_SET_SIZE = 4


# --- Kronfluence task definitions ---


class TextClassificationTask(Task):
    # Partially copied from kronfluence/examples/glue.

    def compute_train_loss(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits

        if not sample:
            return F.cross_entropy(logits, batch["labels"], reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from MadryLab/trak modelout_functions.py.
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits

        labels = batch["labels"]
        bindex = torch.arange(logits.shape[0]).to(
            device=logits.device, non_blocking=False
        )
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

    def get_attention_mask(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return batch["attention_mask"]


class LanguageModelingTask(Task):
    # Copied from kronfluence/examples/wikitext/analyze.py.

    def compute_train_loss(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))

        if not sample:
            labels = batch["labels"]
            labels = labels[..., 1:].contiguous()
            return F.cross_entropy(logits, labels.view(-1), reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        return self.compute_train_loss(batch, model)

    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []
        for i in range(12):
            total_modules.append(f"transformer.h.{i}.attn.c_attn")
            total_modules.append(f"transformer.h.{i}.attn.c_proj")
        for i in range(12):
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")
        return total_modules

    def get_attention_mask(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return batch["attention_mask"]


class LanguageModelingTaskExtended(Task):
    def compute_train_loss(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous()

        if not sample:
            return F.cross_entropy(
                logits, labels.view(-1), reduction="sum", ignore_index=-100
            )
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
            masks = labels.view(-1) == -100
            sampled_labels[masks] = -100
        return F.cross_entropy(
            logits, sampled_labels, ignore_index=-100, reduction="sum"
        )

    def compute_measurement(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(
            logits, shift_labels, ignore_index=-100, reduction="sum"
        )

    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []
        for i in range(12):
            total_modules.append(f"transformer.h.{i}.attn.c_attn")
            total_modules.append(f"transformer.h.{i}.attn.c_proj")
        for i in range(12):
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")
        return total_modules

    def get_attention_mask(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        return batch["attention_mask"]


class DummyLanguageModelingTask(LanguageModelingTask):
    def get_influence_tracked_modules(self) -> List[str]:
        total_modules = []
        for i in range(2):
            total_modules.append(f"transformer.h.{i}.attn.c_attn")
            total_modules.append(f"transformer.h.{i}.attn.c_proj")
        for i in range(2):
            total_modules.append(f"transformer.h.{i}.mlp.c_fc")
            total_modules.append(f"transformer.h.{i}.mlp.c_proj")
        return total_modules


class SimpleLanguageModelingTask(LanguageModelingTask):
    def get_influence_tracked_modules(self) -> List[str]:
        return [
            "mlp1.0",
            "mlp1.2",
            "mlp2.0",
            "mlp2.2",
            "lm_head",
        ]


@pytest.fixture
def text_classification_task():
    return TextClassificationTask()


@pytest.fixture
def language_modeling_task():
    return LanguageModelingTask()


@pytest.fixture
def language_modeling_task_extended():
    return LanguageModelingTaskExtended()


@pytest.fixture
def dummy_language_modeling_task():
    return DummyLanguageModelingTask()


@pytest.fixture
def simple_language_modeling_task() -> SimpleLanguageModelingTask:
    return SimpleLanguageModelingTask()


# --- GPT-2 / Wikitext / fact tracing ---


def replace_conv1d_modules(model: nn.Module) -> None:
    # Partially copied from kronfluence/examples/wikitext/pipeline.py.
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_conv1d_modules(module)

        if module.__class__.__name__ == "Conv1D":
            new_module = nn.Linear(
                in_features=module.weight.shape[0],
                out_features=module.weight.shape[1],
            )
            new_module.weight.data.copy_(module.weight.data.t())
            new_module.bias.data.copy_(module.bias.data)
            setattr(model, name, new_module)


@pytest.fixture
def load_hf_gpt2_trex_finetuned():
    """HF GPT-2 small T-REx-finetuned with Conv1D → Linear swap."""
    from quanda.benchmarks.resources.modules import HFGPT2

    model = HFGPT2.from_pretrained(
        "quanda-bench-test/gpt2-small-trex-openwebtext-ft-hf"
    )
    replace_conv1d_modules(model)
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


@pytest.fixture
def load_fact_tracing_dataset_gpt2_small():
    cfg = {
        "dataset_str": "quanda-bench-test/trex-subset-benchmark",
        "dataset_split": "train",
        "tokenizer": {"backend": "tiktoken", "encoding": "gpt2"},
        "num_prompts": 5,
        "max_evidence_per_prompt": 2,
        "max_length": 64,
        "seed": 42,
    }
    prompt_ds, evidence_ds, entailment_labels, _ = (
        FactTracingConfigParser.parse_fact_tracing_cfg(cfg)
    )
    return prompt_ds, evidence_ds, entailment_labels


@pytest.fixture
def load_gpt2_model():
    config = AutoConfig.from_pretrained("gpt2", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",
        from_tf=False,
        config=config,
        ignore_mismatched_sizes=False,
        trust_remote_code=True,
    )
    replace_conv1d_modules(model)
    return model


@pytest.fixture
def load_wikitext_dataset():
    split = "train"
    indices = [i for i in range(2)]

    raw_datasets = datasets.load_dataset("wikitext", "wikitext-2-raw-v1")
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", use_fast=True, trust_remote_code=True
    )

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    block_size = 16

    def group_texts(examples):
        concatenated_examples = {
            k: list(chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [
                t[i : i + block_size]
                for i in range(0, total_length, block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    if split in ["train", "eval_train"]:
        ds = lm_datasets["train"]
    else:
        ds = lm_datasets["validation"]

    if indices is not None:
        ds = ds.select(indices)

    return ds


# --- GLUE/QNLI ---


def get_glue_dataset(
    data_name: str,
    split: str,
    indices: List[int] = None,
) -> Dataset:
    assert split in ["train", "eval_train", "valid"]

    raw_datasets = datasets.load_dataset(path="glue", name=data_name)
    label_list = raw_datasets["train"].features["label"].names
    num_labels = len(label_list)
    assert num_labels == 2

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-base-cased", use_fast=True, trust_remote_code=True
    )

    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS[data_name]
    padding = "max_length"
    max_seq_length = 128

    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *texts, padding=padding, max_length=max_seq_length, truncation=True
        )
        if "label" in examples:
            result["labels"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
    )

    if split in ["train", "eval_train"]:
        ds = raw_datasets["train"]
        if data_name == "rte":
            ds = ds.select(range(2432))
    else:
        ds = raw_datasets["validation"]

    if indices is not None:
        ds = ds.select(indices)

    return ds


def _get_qnli_dataset(split, inds=None):
    raw_datasets = datasets.load_dataset(
        "glue",
        "qnli",
        cache_dir=None,
        use_auth_token=None,
    )
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS["qnli"]

    tokenizer = AutoTokenizer.from_pretrained(
        "gchhablani/bert-base-cased-finetuned-qnli",
        cache_dir=None,
        use_fast=True,
        revision="main",
        token=None,
    )

    padding = "max_length"
    max_seq_length = 128

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )
        result["labels"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    if split == "train":
        return raw_datasets["train"]
    return raw_datasets["validation"]


@pytest.fixture
def load_qnli_model():
    return SequenceClassificationModel()


@pytest.fixture
def load_qnli_dataset():
    ds_train = _get_qnli_dataset("train").select(range(QNLI_TRAIN_SET_SIZE))
    ds_val = _get_qnli_dataset("validation").select(range(QNLI_VAL_SET_SIZE))
    return ds_train, ds_val


# --- Simple text classifier / dummy text data ---


@pytest.fixture
def load_simple_classifier():
    return SimpleTextClassifier()


@pytest.fixture
def load_text_dataset():
    def create_dummy_data(size, is_train=True):
        seq_length = 10

        if is_train:
            base_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            input_ids = [base_ids for _ in range(size)]
            labels = [i % 2 for i in range(size)]
        else:
            input_ids = [[10, 9, 8, 7, 6, 5, 4, 3, 2, 1] for _ in range(size)]
            labels = [0, 1, 0, 1, 0][:size]

        attention_mask = [[1] * seq_length for _ in range(size)]
        token_type_ids = [[0] * seq_length for _ in range(size)]

        return datasets.Dataset.from_dict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "labels": labels,
            }
        )

    ds_train = create_dummy_data(20, is_train=True)
    ds_val = create_dummy_data(5, is_train=False)
    return ds_train, ds_val


# --- Causal LM dummy/test fixtures ---


@pytest.fixture
def load_dummy_causal_lm_model():
    return TinyGPT2()


@pytest.fixture
def load_dummy_causal_lm_dataset():
    vocab_size = 100
    seq_length = 16
    num_samples = 5

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(num_samples, seq_length),
        dtype=torch.long,
    ).tolist()

    attention_mask = [[1] * seq_length for _ in range(num_samples)]
    labels = [ids.copy() for ids in input_ids]

    return datasets.Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    )


@pytest.fixture
def causal_lm_test_dataset():
    vocab_size = 100
    seq_length = 16
    num_queries = 3

    np.random.seed(42)
    input_ids = np.random.randint(
        low=0, high=vocab_size, size=(num_queries, seq_length), dtype=np.int64
    )
    attention_mask = np.ones((num_queries, seq_length), dtype=np.int64)
    labels = input_ids.copy()

    return datasets.Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    )


@pytest.fixture
def causal_lm_test_entailment_labels():
    num_queries = 3
    num_training_examples = 10

    entailment_labels = torch.zeros(
        (num_queries, num_training_examples), dtype=torch.bool
    )
    entailment_labels[0, 1] = True
    entailment_labels[1, 0] = True
    entailment_labels[2, 2] = True
    return entailment_labels


@pytest.fixture
def load_fact_tracing_dataset():
    """Build prompt/evidence/entailment via the fact-tracing parser (HF GPT-2)."""
    cfg = {
        "dataset_str": "quanda-bench-test/trex-subset-benchmark",
        "dataset_split": "train",
        "tokenizer": {"backend": "hf", "name": "gpt2"},
        "num_prompts": 2,
        "max_evidence_per_prompt": 3,
        "max_length": 64,
        "seed": 42,
    }
    prompt_ds, evidence_ds, entailment_labels, _ = (
        FactTracingConfigParser.parse_fact_tracing_cfg(cfg)
    )
    return prompt_ds, evidence_ds, entailment_labels


@pytest.fixture
def load_simple_causal_lm_model() -> SimpleCausalLM:
    return SimpleCausalLM()


@pytest.fixture
def load_simple_causal_lm_dataset() -> datasets.Dataset:
    input_ids = torch.randint(0, 100, (5, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    return datasets.Dataset.from_dict(
        {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "labels": labels.tolist(),
        }
    )
