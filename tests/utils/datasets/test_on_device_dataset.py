import pytest
import torch

from quanda.utils.datasets.on_device_dataset import (
    OnDeviceDataset,
    _move_item_to_device,
)


@pytest.mark.utils
def test_move_item_handles_tensor_tuple_list_dict_and_scalars():
    device = torch.device("cpu")
    item = {
        "x": torch.tensor([1.0, 2.0]),
        "meta": ("tag", [torch.tensor([3]), 4]),
        "flag": True,
        "score": 1.5,
    }

    moved = _move_item_to_device(item, device)

    assert moved["x"].device == device
    assert moved["meta"][0] == "tag"
    assert moved["meta"][1][0].device == device
    assert moved["meta"][1][1].item() == 4
    assert moved["flag"].dtype == torch.bool
    assert moved["score"].item() == 1.5


@pytest.mark.utils
def test_move_item_returns_non_tensor_leaf_unchanged():
    sentinel = object()
    assert _move_item_to_device(sentinel, "cpu") is sentinel


@pytest.mark.utils
def test_on_device_dataset_moves_tensor_dataset_samples():
    base = torch.utils.data.TensorDataset(
        torch.arange(6.0).view(3, 2),
        torch.tensor([0, 1, 0]),
    )
    wrapped = OnDeviceDataset(base, device="cpu")

    sample_x, sample_y = wrapped[1]

    assert torch.equal(sample_x, torch.tensor([2.0, 3.0]))
    assert sample_y.item() == 1
    assert len(wrapped) == 3
