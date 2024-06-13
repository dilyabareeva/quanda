import functools
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from tests.models import LeNet

MNIST_IMAGE_SIZE = 28
BATCH_SIZE = 124
MINI_BATCH_SIZE = 8
RANDOM_SEED = 42


@pytest.fixture()
def load_dataset():
    x = torch.stack([torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2)])
    y = torch.tensor([0, 1, 0]).long()
    return torch.utils.data.TensorDataset(x, y)


@pytest.fixture()
def load_rand_tensor():
    return torch.rand(10, 10).float()


@pytest.fixture()
def load_rand_test_predictions():
    return torch.randint(0, 10, (10000,))


@pytest.fixture()
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle))
    return model


@pytest.fixture()
def load_init_mnist_model():
    """Load a not trained LeNet classification model (architecture at quantus/helpers/models)."""
    return LeNet()


@pytest.fixture()
def load_mnist_dataset():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = (
        np.loadtxt("tests/assets/mnist_test_suite_1/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_test_suite_1/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TensorDataset(torch.tensor(x_batch).float(), torch.tensor(y_batch).long())
    return dataset


@pytest.fixture()
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


@pytest.fixture()
def load_mnist_test_samples_1():
    return torch.load("tests/assets/mnist_test_suite_1/test_dataset.pt")


@pytest.fixture()
def load_mnist_test_labels_1():
    return torch.load("tests/assets/mnist_test_suite_1/test_labels.pt")


@pytest.fixture()
def load_mnist_explanations_1():
    return torch.load("tests/assets/mnist_test_suite_1/mnist_SimilarityInfluence_tda.pt")


@pytest.fixture()
def torch_cross_entropy_loss_object():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture()
def torch_sgd_optimizer():
    return functools.partial(torch.optim.SGD, lr=0.01, momentum=0.9)
