import json
import os
import pickle
from typing import Dict, List, Tuple

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
from transformers import AutoTokenizer

from quanda.benchmarks.resources import config_map
from quanda.utils.datasets.transformed.label_flipping import (
    LabelFlippingDataset,
)
from quanda.utils.datasets.transformed.label_grouping import (
    LabelGroupingDataset,
)
from quanda.utils.training import Trainer
from quanda.utils.training.base_pl_module import BasicLightningModule
from tests.models import (
    LeNet,
    SequenceClassificationModel,
    SimpleTextClassifier,
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
    metadata = LabelGroupingDataset.metadata_cls(
        seed=27,
        n_groups=2,
        class_to_group="random",
    )
    return LabelGroupingDataset(
        dataset,
        metadata=metadata,
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


class ClassificationTask(Task):
    # Copied from: https://github.com/pomonam/kronfluence/blob/main/examples/cifar/analyze.py
    def compute_train_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
        inputs, labels = batch
        logits = model(inputs)

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


@pytest.fixture
def classification_task():
    return ClassificationTask()


# Partially copied from https://github.com/pomonam/kronfluence/tree/main/examples/glue.
class TextClassificationTask(Task):
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
        "tests/assets/mnist_local_bench/20fba38-default_ClassDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
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
        "tests/assets/mnist_local_bench/20fba38-default_LDS.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)

    return config


@pytest.fixture
def load_mnist_mislabeling_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/20fba38-default_MislabelingDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def load_mnist_subclass_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/20fba38-default_SubclassDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def load_mnist_shortcut_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/20fba38-default_ShortcutDetection.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def load_mnist_mixed_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/20fba38-default_MixedDatasets.yaml",
        "r",
    ) as f:
        config = yaml.safe_load(f)
    return config


@pytest.fixture
def load_mnist_lds_config():
    # load yaml file
    with open(
        "tests/assets/mnist_local_bench/20fba38-default_LDS.yaml",
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
