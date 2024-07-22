import json
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.utils.datasets.transformed.label_flipping import LabelFlippingDataset
from src.utils.datasets.transformed.label_grouping import LabelGroupingDataset
from tests.models import LeNet

MNIST_IMAGE_SIZE = 28
BATCH_SIZE = 124
MINI_BATCH_SIZE = 8
RANDOM_SEED = 42


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
        [[i * 1.0 for i in range(8)], [i * 1.0 for i in range(8)], [i * 1.0 for i in range(8)]], dtype=torch.float
    )


@pytest.fixture
def range_ranking():
    return torch.tensor([i for i in range(8)])


@pytest.fixture
def mnist_seed_27_poisoned_labels():
    with open("tests/assets/mnist_seed_27_poisoned_labels.json", "r") as f:
        return json.load(f)


@pytest.fixture
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle))
    return model


@pytest.fixture
def load_init_mnist_model():
    """Load a not trained LeNet classification model (architecture at quantus/helpers/models)."""
    return LeNet()


@pytest.fixture
def load_mnist_dataset():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = (
        np.loadtxt("tests/assets/mnist_test_suite_1/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_test_suite_1/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(torch.tensor(x_batch).float(), torch.tensor(y_batch).long())
    return dataset


@pytest.fixture
def load_mnist_labels():
    y_batch = np.loadtxt("tests/assets/mnist_test_suite_1/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    return torch.tensor(y_batch).long()


@pytest.fixture
def load_grouped_mnist_dataset():
    x_batch = (
        np.loadtxt("tests/assets/mnist_test_suite_1/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_test_suite_1/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(torch.tensor(x_batch).float(), torch.tensor(y_batch).long())
    return LabelGroupingDataset(
        dataset,
        n_classes=10,
        n_groups=2,
        class_to_group="random",
        seed=27,
        device="cpu",
    )


@pytest.fixture
def load_poisoned_mnist_dataset():
    x_batch = (
        np.loadtxt("tests/assets/mnist_test_suite_1/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_test_suite_1/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(torch.tensor(x_batch).float(), torch.tensor(y_batch).long())
    return LabelFlippingDataset(
        dataset,
        n_classes=10,
        p=1.0,
        seed=27,
        device="cpu",
    )


@pytest.fixture
def load_mnist_dataloader():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = (
        np.loadtxt("tests/assets/mnist_test_suite_1/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_test_suite_1/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TensorDataset(torch.tensor(x_batch).float(), torch.tensor(y_batch).long())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=MINI_BATCH_SIZE, shuffle=False)
    return dataloader


@pytest.fixture
def load_mnist_test_samples_1():
    return torch.load("tests/assets/mnist_test_suite_1/test_dataset.pt")


@pytest.fixture
def load_mnist_test_labels_1():
    return torch.load("tests/assets/mnist_test_suite_1/test_labels.pt")


@pytest.fixture
def load_mnist_explanations_1():
    return torch.load("tests/assets/mnist_test_suite_1/mnist_SimilarityInfluence_tda.pt")


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
