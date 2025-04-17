import json
import os
import pickle
import random
from itertools import chain
from typing import Dict, List

import datasets
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from kronfluence.task import Task  # type: ignore
from torch.utils.data import Dataset, TensorDataset
from torchvision.models import resnet18, vit_b_16
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from quanda.benchmarks.resources import config_map
from quanda.utils.datasets.transformed.label_flipping import (
    LabelFlippingDataset,
)
from quanda.utils.datasets.transformed.label_grouping import (
    LabelGroupingDataset,
)
from quanda.utils.datasets.transformed.metadata import ClassMapping
from quanda.utils.training import Trainer
from quanda.utils.training.base_pl_module import BasicLightningModule
from tests.models import (
    LeNet,
    SequenceClassificationModel,
    SimpleTextClassifier,
    TinyGPT2
)

# Copied from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py.
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
MNIST_IMAGE_SIZE = 28
BATCH_SIZE = 124
MINI_BATCH_SIZE = 8
RANDOM_SEED = 42


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "local: only run this test if running locally"
    )


def pytest_runtest_setup(item):
    if "local" in item.keywords and os.getenv("GITHUB_ACTIONS"):
        pytest.skip("Skipping local-only tests on GitHub Actions")


class TestTensorDataset(TensorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = self.tensors[0]
        self._targets = self.tensors[1]

    def __getitem__(self, index):
        return self._data[index], self._targets[index].item()


@pytest.fixture
def load_dataset():
    x = torch.stack([torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2)])
    y = torch.tensor([0, 1, 0]).long()
    return torch.utils.data.TensorDataset(x, y)


@pytest.fixture
def load_rand_tensor():
    return torch.rand(10, 10).float()


@pytest.fixture
def load_rand_test_predictions():
    return torch.randint(0, 10, (10000,))


@pytest.fixture
def mnist_range_explanations():
    return torch.tensor(
        [
            [i * 1.0 for i in range(8)],
            [i * 1.0 for i in range(8)],
            [i * 1.0 for i in range(8)],
        ],
        dtype=torch.float,
    )


@pytest.fixture
def range_ranking():
    return torch.tensor([i for i in range(8)])


@pytest.fixture
def mnist_seed_27_mislabeling_labels():
    with open(
        "tests/assets/dataset/mnist_seed_27_poisoned_labels.json",
        "r",
    ) as f:
        return json.load(f)


@pytest.fixture
def load_mnist_checkpoints():
    """Get paths to five checkpoints from a single training run."""
    checkpoint_paths = [
        "tests/assets/mnist_checkpoints/checkpoint-00",
        "tests/assets/mnist_checkpoints/checkpoint-01",
        "tests/assets/mnist_checkpoints/checkpoint-02",
        "tests/assets/mnist_checkpoints/checkpoint-03",
        "tests/assets/mnist_checkpoints/checkpoint-04",
    ]
    return checkpoint_paths


@pytest.fixture
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load(
            "tests/assets/mnist", map_location="cpu", pickle_module=pickle
        )
    )
    return model


@pytest.fixture
def load_mnist_model_with_custom_param():
    """Load a pre-trained LeNet classification model with a custom parameter
    (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load(
            "tests/assets/mnist", map_location="cpu", pickle_module=pickle
        )
    )
    model.custom_param = torch.nn.Parameter(torch.randn(4))
    return model


@pytest.fixture
def load_mnist_last_checkpoint():
    """Load the path to the last checkpoint of a pre-trained LeNet classification model."""
    return "tests/assets/mnist"


@pytest.fixture
def load_mnist_pl_module():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load(
            "tests/assets/mnist", map_location="cpu", pickle_module=pickle
        )
    )

    pl_module = BasicLightningModule(
        model=model,
        optimizer=torch.optim.SGD,
        lr=0.01,
        criterion=torch.nn.CrossEntropyLoss(),
    )

    return pl_module


@pytest.fixture
def load_mnist_grouped_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet(num_outputs=2)
    model.load_state_dict(
        torch.load(
            "tests/assets/mnist_grouped_model",
            map_location="cpu",
            pickle_module=pickle,
        )
    )
    return model


@pytest.fixture
def load_init_mnist_model():
    """Load a not trained LeNet classification model (architecture at quantus/helpers/models)."""
    return LeNet()


@pytest.fixture
def load_mnist_dataset():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = (
        np.loadtxt("tests/assets/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    return dataset


@pytest.fixture
def load_mnist_labels():
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    return torch.tensor(y_batch).long()


@pytest.fixture
def load_mnist_adversarial_indices():
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    return [int(y == 1) for y in y_batch]


@pytest.fixture
def load_grouped_mnist_dataset():
    x_batch = (
        np.loadtxt("tests/assets/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    metadata = LabelGroupingDataset.metadata_cls(seed=27)
    mapping = ClassMapping(
        class_to_group=ClassMapping._generate(
            n_classes=10, n_groups=2, seed=27
        ),
        n_classes=10,
        n_groups=2,
        seed=27,
    )
    return LabelGroupingDataset(
        dataset,
        metadata=metadata,
        class_to_group=mapping.class_to_group,
        n_classes=mapping.n_classes,
        n_groups=mapping.n_groups,
    )


@pytest.fixture
def load_mislabeling_mnist_dataset():
    x_batch = (
        np.loadtxt("tests/assets/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    metadata = LabelFlippingDataset.metadata_cls(
        p=1.0,
        seed=27,
    )
    return LabelFlippingDataset(
        dataset,
        metadata=metadata,
    )


@pytest.fixture
def load_mnist_dataloader():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = (
        np.loadtxt("tests/assets/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=MINI_BATCH_SIZE, shuffle=False
    )
    return dataloader


@pytest.fixture
def load_mnist_test_samples_1():
    return torch.load("tests/assets/dataset/test_dataset.pt")


@pytest.fixture
def load_mnist_test_samples_batches():
    return [
        torch.load("tests/assets/dataset/test_dataset.pt"),
        torch.load("tests/assets/dataset/test_dataset_2.pt"),
    ]


@pytest.fixture
def load_mnist_test_labels_1():
    return torch.load("tests/assets/dataset/test_labels.pt")


@pytest.fixture
def load_mnist_test_labels_1_list():
    return torch.load("tests/assets/dataset/test_labels.pt").tolist()


@pytest.fixture
def load_mnist_explanations_similarity_1():
    return torch.load("tests/assets/tda/mnist_SimilarityInfluence_tda.pt")


@pytest.fixture
def load_mnist_explanations_dot_similarity_1():
    return torch.load("tests/assets/tda/mnist_SimilarityInfluence_dot_tda.pt")


@pytest.fixture
def load_mnist_dataset_explanations():
    return torch.rand((MINI_BATCH_SIZE, MINI_BATCH_SIZE))


@pytest.fixture
def torch_cross_entropy_loss_object():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def torch_constant_lr_scheduler_type():
    return torch.optim.lr_scheduler.ConstantLR


@pytest.fixture
def torch_sgd_optimizer():
    return torch.optim.SGD


@pytest.fixture
def load_fashion_mnist_to_mnist_transform():
    # change tp black and white

    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ]
    )


@pytest.fixture
def load_fashion_mnist_path():
    return "tests/assets/fashion_mnist_examples"


@pytest.fixture
def mnist_white_square_transformation():
    def add_white_square(img):
        img[:, 8:13, 10:15] = (
            1.0  # Paste it onto the image at the specified position
        )
        return img

    return add_white_square


@pytest.fixture
def load_subset_indices_lds():
    return "subset_indices.yaml"


@pytest.fixture
def load_pretrained_models_lds():
    return [
        "tests/assets/lds_checkpoints/model_subset_0.pt",
        "tests/assets/lds_checkpoints/model_subset_1.pt",
        "tests/assets/lds_checkpoints/model_subset_2.pt",
        "tests/assets/lds_checkpoints/model_subset_3.pt",
    ]


@pytest.fixture
def load_vit():
    return vit_b_16()


@pytest.fixture
def load_resnet():
    return resnet18()


@pytest.fixture
def classification_task():
    from quanda.explainers.wrappers.kronfluence_tasks import (
        ImageClassificationTask,
    )

    return ImageClassificationTask()


class TextClassificationTask(Task):
    # Partially copied from: https://github.com/pomonam/kronfluence/tree/main/examples/glue.
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
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
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


@pytest.fixture
def text_classification_task():
    return TextClassificationTask()


class LanguageModelingTask(Task):
    # Copied from: https://github.com/pomonam/kronfluence/blob/main/examples/wikitext/analyze.py
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
            summed_loss = F.cross_entropy(
                logits, labels.view(-1), reduction="sum"
            )
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
            summed_loss = F.cross_entropy(
                logits, sampled_labels, reduction="sum"
            )
        return summed_loss

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


@pytest.fixture
def language_modeling_task():
    return LanguageModelingTask()


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
            summed_loss = F.cross_entropy(
                logits, labels.view(-1), reduction="sum", ignore_index=-100
            )
        else:
            with torch.no_grad():
                probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
                sampled_labels = torch.multinomial(
                    probs,
                    num_samples=1,
                ).flatten()
                masks = labels.view(-1) == -100
                sampled_labels[masks] = -100
            summed_loss = F.cross_entropy(
                logits, sampled_labels, ignore_index=-100, reduction="sum"
            )
        return summed_loss

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


@pytest.fixture
def language_modeling_task_extended():
    return LanguageModelingTaskExtended()


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


@pytest.fixture
def dummy_language_modeling_task():
    return DummyLanguageModelingTask()


def replace_conv1d_modules(model: nn.Module) -> None:
    # Partially copied from https://github.com/pomonam/kronfluence/blob/main/examples/wikitext/pipeline.py
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
def load_gpt2_model():
    config = AutoConfig.from_pretrained(
        "gpt2",
        trust_remote_code=True,
    )
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
        train_dataset = lm_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = lm_datasets["validation"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return ds


def get_glue_dataset(
    data_name: str,
    split: str,
    indices: List[int] = None,
) -> Dataset:
    assert split in ["train", "eval_train", "valid"]

    raw_datasets = datasets.load_dataset(
        path="glue",
        name=data_name,
    )
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
        train_dataset = raw_datasets["train"]
        ds = train_dataset
        if data_name == "rte":
            ds = ds.select(range(2432))
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return ds


class SequenceClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            "gchhablani/bert-base-cased-finetuned-qnli",
            num_labels=2,
            finetuning_task="qnli",
            cache_dir=None,
            revision="main",
            token=None,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "gchhablani/bert-base-cased-finetuned-qnli",
            config=self.config,
            cache_dir=None,
            revision="main",
            token=None,
            ignore_mismatched_sizes=False,
        )

        self.model.eval()

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )


def get_dataset(split, inds=None):
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
        load_from_cache_file=(not False),
        desc="Running tokenizer on dataset",
    )

    if split == "train":
        train_dataset = raw_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset
    return ds


@pytest.fixture
def load_qnli_model():
    model = SequenceClassificationModel()
    return model


@pytest.fixture
def load_qnli_dataset():
    ds_train = get_dataset("train")
    ds_train = ds_train.select(range(QNLI_TRAIN_SET_SIZE))
    ds_val = get_dataset("validation")
    ds_val = ds_val.select(range(QNLI_VAL_SET_SIZE))
    return ds_train, ds_val


@pytest.fixture
def load_mnist_unit_test_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/83edb41-default_ClassDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def load_mnist_unit_test_config_one_cycle():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/83edb41-default_ClassDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    config["model"]["trainer"]["scheduler"] = "one_cycle"
    config["model"]["trainer"]["scheduler_kwargs"] = {
        "max_lr": 0.02,
        "interval": "step",
    }
    return config


@pytest.fixture
def load_mnist_unit_test_config_num_ckpts_2():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/83edb41-default_ClassDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    config["num_checkpoints"] = 2
    return config


@pytest.fixture
def load_mnist_unit_test_config_hf():
    # load yaml file
    with open(
        config_map["mnist_class_detection_unit"],
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def load_mnist_linear_datamodeling_config(
    load_pretrained_models_lds, load_subset_indices_lds
):
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/83edb41-default_LDS.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)

    return config


@pytest.fixture
def load_mnist_mislabeling_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/83edb41-default_MislabelingDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def load_mnist_subclass_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/83edb41-default_SubclassDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def load_mnist_shortcut_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/83edb41-default_ShortcutDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def load_mnist_mixed_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/83edb41-default_MixedDatasets.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def load_mnist_lds_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/83edb41-default_LDS.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


def load_yaml(file_path):
    """Helper function to load a YAML file as a Python dictionary."""
    assert os.path.exists(file_path), f"Config file not found: {file_path}"
    with open(file_path, "r") as f:
        return yaml.safe_load(f)  # Load YAML as Python dict


@pytest.fixture
def load_wandb_config():
    """Load WandB config from wandb.yaml as a Python dict."""
    dict = load_yaml("config/logger/wandb.yaml")
    dict["offline"] = True
    return dict


@pytest.fixture
def load_tensorboard_config():
    """Load TensorBoard config from tensorboard.yaml as a Python dict."""
    return load_yaml("config/logger/tensorboard.yaml")


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

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

        return datasets.Dataset.from_dict(data)

    ds_train = create_dummy_data(20, is_train=True)
    ds_val = create_dummy_data(5, is_train=False)

    return ds_train, ds_val


@pytest.fixture
def dummy_trainer():
    return Trainer(
        max_epochs=3,
        optimizer=torch.optim.SGD,
        lr=0.1,
        criterion=torch.nn.CrossEntropyLoss(),
    )


@pytest.fixture
def two_field_dataset():
    def _create(n=8):
        return datasets.Dataset.from_dict(
            {
                "question": [f"What is {i}?" for i in range(n)],
                "sentence": [f"Answer {i}" for i in range(n)],
                "label": [i % 2 for i in range(n)],
            }
        )

    return _create


@pytest.fixture
def single_field_dataset():
    def _create():
        return datasets.Dataset.from_dict(
            {"text": ["Hello world", "Foo bar"], "label": [0, 1]}
        )

    return _create


@pytest.fixture
def custom_label_dataset():
    def _create():
        return datasets.Dataset.from_dict(
            {"text": ["a", "b", "c"], "my_label": [0, 1, 2]}
        )

    return _create


class _LogitsOutput:
    """Wraps a tensor to expose a .logits attribute."""

    def __init__(self, logits):
        self.logits = logits


class _ConstantModel(torch.nn.Module):
    """Model that always predicts a fixed class."""

    def __init__(self, predicted_class, num_classes):
        super().__init__()
        self.predicted_class = predicted_class
        self.num_classes = num_classes

    def forward(self, x):
        bs = x.shape[0]
        logits = torch.zeros(bs, self.num_classes)
        logits[:, self.predicted_class] = 10.0
        return logits


class _ConstantDictModel(torch.nn.Module):
    """Dict-input model that always predicts a class."""

    def __init__(self, predicted_class, num_classes, wrap_logits):
        super().__init__()
        self.predicted_class = predicted_class
        self.num_classes = num_classes
        self.wrap_logits = wrap_logits

    def forward(self, **kwargs):
        first_val = next(iter(kwargs.values()))
        bs = first_val.shape[0]
        logits = torch.zeros(bs, self.num_classes)
        logits[:, self.predicted_class] = 10.0
        if self.wrap_logits:
            return _LogitsOutput(logits)
        return logits


class _DictDataset(Dataset):
    """Dataset that yields dict batches."""

    def __init__(self, data_dict):
        self.data = data_dict
        self.n = len(next(iter(data_dict.values())))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


@pytest.fixture
def constant_model():
    """Factory: model always predicting a given class."""

    def _create(predicted_class=0, num_classes=3):
        model = _ConstantModel(predicted_class, num_classes)
        model.eval()
        return model

    return _create


@pytest.fixture
def constant_dict_model():
    """Factory: dict-input model predicting a class."""

    def _create(
        predicted_class=0,
        num_classes=3,
        wrap_logits=True,
    ):
        model = _ConstantDictModel(predicted_class, num_classes, wrap_logits)
        model.eval()
        return model

    return _create


@pytest.fixture
def tuple_dataloader():
    """Factory: DataLoader of (tensor, label) tuples."""

    def _create(labels, n_features=4, batch_size=4):
        n = len(labels)
        x = torch.rand(n, n_features)
        y = torch.tensor(labels, dtype=torch.long)
        ds = TensorDataset(x, y)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size)

    return _create


@pytest.fixture
def dict_dataloader():
    """Factory: DataLoader yielding dict batches."""

    def _create(labels, seq_length=8, batch_size=4):
        n = len(labels)
        data = {
            "input_ids": torch.randint(0, 100, (n, seq_length)),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
        ds = _DictDataset(data)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size)

    return _create


@pytest.fixture
def load_dummy_causal_lm_model():
    model = TinyGPT2()
    return model


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

    data = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    dataset = datasets.Dataset.from_dict(data)
    return dataset


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

    data_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

    return datasets.Dataset.from_dict(data_dict)


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
def load_fact_tracing_dataset(
    dataset_name="quanda-bench-test/trex-subset",
    tokenizer_name="gpt2",
    num_prompts=3,
    max_evidence_per_prompt=3,
    max_length=64,
    seed=42,
):
    # Load GPT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load the TREx dataset
    dataset = datasets.load_dataset(dataset_name, split="train")

    # Sample prompt entries
    if num_prompts < len(dataset):
        random.seed(seed)
        indices = random.sample(range(len(dataset)), num_prompts)
        sampled_dataset = dataset.select(indices)
    else:
        sampled_dataset = dataset

    # Tokenize prompt + answer and mask prompt in labels
    input_ids = []
    attention_mask = []
    labels = []
    prompts = []
    answers = []

    for entry in sampled_dataset:
        prompt = entry["prompt"].strip()
        answer = entry["answer"][0].strip()
        full_text = prompt + " " + answer

        encoded = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length)[
            "input_ids"
        ]
        prompt_len = len(prompt_ids)

        label_ids = encoded["input_ids"].copy()
        label_ids[:prompt_len] = [
            -100
        ] * prompt_len  # Mask out prompt from loss

        input_ids.append(encoded["input_ids"])
        attention_mask.append(encoded["attention_mask"])
        labels.append(label_ids)
        prompts.append(prompt)
        answers.append(answer)

    prompt_dataset = datasets.Dataset.from_dict(
        {
            "prompt": prompts,
            "answer": answers,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    )

    # Gather evidence sentences
    evidence_sentences = []
    evidence_map = []

    for i, entry in enumerate(sampled_dataset):
        selected = entry["evidence_sentences"][:max_evidence_per_prompt]
        for sentence in selected:
            evidence_sentences.append(sentence)
            evidence_map.append(i)

    # Tokenize evidence sentences
    evidence_input_ids = []
    evidence_attention_mask = []
    for sentence in evidence_sentences:
        encoded = tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        evidence_input_ids.append(encoded["input_ids"])
        evidence_attention_mask.append(encoded["attention_mask"])

    evidence_dataset = datasets.Dataset.from_dict(
        {
            "sentence": evidence_sentences,
            "input_ids": evidence_input_ids,
            "attention_mask": evidence_attention_mask,
            "labels": evidence_input_ids,
        }
    )

    # Create entailment matrix
    num_queries = len(prompt_dataset)
    num_evidence = len(evidence_dataset)
    entailment_labels = torch.zeros(
        (num_queries, num_evidence), dtype=torch.long
    )
    for i in range(num_queries):
        for j in range(num_evidence):
            if evidence_map[j] == i:
                entailment_labels[i, j] = 1

    return prompt_dataset, evidence_dataset, entailment_labels
