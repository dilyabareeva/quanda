import pytest
import torch


@pytest.fixture()
def load_dataset():
    x = torch.stack([torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2)])
    y = torch.tensor([0, 1, 0]).long()
    return torch.utils.data.TensorDataset(x, y)


@pytest.fixture()
def load_rand_tensor_explanations():
    return torch.rand(10000, 10)


@pytest.fixture()
def load_rand_test_predictions():
    return torch.randint(0, 10, (10000,))
