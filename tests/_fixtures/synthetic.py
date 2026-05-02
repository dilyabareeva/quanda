"""Synthetic / generic test fixtures: rand tensors, dummy models, factories."""

import os

import datasets
import pytest
import torch
import yaml
from torch.utils.data import Dataset, TensorDataset
from torchvision.models import resnet18, vit_b_16

from quanda.utils.training import Trainer


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
def torch_cross_entropy_loss_object():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def torch_constant_lr_scheduler_type():
    return torch.optim.lr_scheduler.ConstantLR


@pytest.fixture
def torch_sgd_optimizer():
    return torch.optim.SGD


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


def _load_yaml(file_path: str) -> dict:
    assert os.path.exists(file_path), f"Config file not found: {file_path}"
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def load_wandb_config():
    cfg = _load_yaml("config/logger/wandb.yaml")
    cfg["offline"] = True
    return cfg


@pytest.fixture
def load_tensorboard_config():
    return _load_yaml("config/logger/tensorboard.yaml")


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
