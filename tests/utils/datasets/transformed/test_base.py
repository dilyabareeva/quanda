import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from src.utils.datasets.transformed.base import TransformedDataset


class UnsizedTensorDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.X = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.X[index], self.y[index])


@pytest.mark.utils
@pytest.mark.parametrize(
    "sized, length",
    [
        (
            True,
            10,
        ),
        # (False, 10),
    ],
)
def test_base_len(
    sized,
    length,
    request,
):
    if sized:
        dataset = TensorDataset(torch.ones((length, 100)), torch.ones((length,)))
    else:
        dataset = UnsizedTensorDataset(torch.ones((length, 100)), torch.ones((length,)))
    dataset = TransformedDataset(dataset=dataset, n_classes=2)
    assert len(dataset) == length
