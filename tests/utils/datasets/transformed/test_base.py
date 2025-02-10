import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from quanda.utils.datasets.transformed import TransformedDataset
from quanda.utils.datasets.transformed.metadata import LabelFlippingMetadata


class UnsizedTensorDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.X = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.X[index], self.y[index])


@pytest.mark.utils
@pytest.mark.parametrize(
    "sized, length, err",
    [
        (True, 10, None),
        (False, 10, ValueError),
    ],
)
def test_transformed_dataset_len(
    sized,
    length,
    err,
    request,
):
    metadata = LabelFlippingMetadata(
        cls_idx=0,
        p=0.5,
    )

    if sized:
        dataset = TensorDataset(
            torch.ones((length, 100)), torch.ones((length,))
        )
        dataset = TransformedDataset(dataset=dataset, metadata=metadata)
        assert len(dataset) == length
    else:
        dataset = UnsizedTensorDataset(
            torch.ones((length, 100)), torch.ones((length,))
        )
        with pytest.raises(err):
            dataset = TransformedDataset(dataset=dataset, metadata=metadata)


@pytest.mark.utils
@pytest.mark.parametrize(
    "dataset, n_classes, sample_fn, label_fn",
    [
        ("load_mnist_dataset", 10, lambda x: torch.zeros_like(x), lambda x: 0),
    ],
)
def test_transformed_dataset(dataset, n_classes, sample_fn, label_fn, request):
    dataset = request.getfixturevalue(dataset)
    metadata = LabelFlippingMetadata()
    metadata.generate_indices(dataset)
    trans_ds = TransformedDataset(
        dataset=dataset,
        sample_fn=sample_fn,
        label_fn=label_fn,
        metadata=metadata,
    )
    cond1 = torch.all(trans_ds[0][0] == 0.0)
    cond2 = trans_ds[0][1] == 0.0
    cond3 = not torch.allclose(trans_ds[0][0], dataset[0][0])
    final_cond = cond1 and cond2 and cond3
    assert final_cond
