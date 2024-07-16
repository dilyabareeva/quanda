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
        (True, 10, None),
        (False, 10, ValueError),
    ],
)
def test_base_len(
    sized,
    length,
    err,
    request,
):
    if sized:
        dataset = TensorDataset(torch.ones((length, 100)), torch.ones((length,)))
        dataset = TransformedDataset(dataset=dataset, n_classes=2)
        assert len(dataset) == length
    else:
        dataset = UnsizedTensorDataset(torch.ones((length, 100)), torch.ones((length,)))
        with pytest.raises(err):
            dataset = TransformedDataset(dataset=dataset, n_classes=2)


@pytest.mark.utils
@pytest.mark.parametrize(
    "dataset, n_classes",
    [
        ("load_mnist_dataset", 10),
    ],
)
def test_base_transformed(dataset, n_classes, request):
    dataset = request.getfixturevalue(dataset)
    trans_ds = TransformedDataset(
        dataset=dataset, n_classes=n_classes, sample_fn=lambda x: torch.zeros_like(x), label_fn=lambda x: 0
    )
    assert torch.all(trans_ds[0][0] == 0.0)
    assert trans_ds[0][1] == 0.0
    assert not torch.allclose(trans_ds[0][0], dataset[0][0])
