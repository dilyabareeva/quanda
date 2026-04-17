import datasets
import pytest
import torch

from quanda.utils.datasets.dataset_handlers import (
    HuggingFaceDatasetHandler,
    TorchDatasetHandler,
    get_dataset_handler,
)


@pytest.mark.utils
@pytest.mark.parametrize(
    "label, expected_type",
    [
        (3, int),
        (torch.tensor(2, dtype=torch.long), torch.Tensor),
    ],
)
def test_huggingface_handler_with_label(label, expected_type):
    handler = HuggingFaceDatasetHandler()
    item = {"input_ids": [1, 2, 3], "labels": label}

    assert handler.get_label(item) == label

    new_item = handler.with_label(item, 5)

    assert new_item is not item
    assert isinstance(new_item["labels"], expected_type)
    if expected_type is torch.Tensor:
        assert new_item["labels"].item() == 5
        assert new_item["labels"].dtype == label.dtype
    else:
        assert new_item["labels"] == 5
    assert item["labels"] == label


@pytest.mark.utils
def test_torch_handler_with_label():
    handler = TorchDatasetHandler()
    sample = torch.rand(2, 2)
    item = (sample, 3)

    assert handler.get_label(item) == 3

    new_item = handler.with_label(item, 7)
    assert new_item[1] == 7
    assert new_item[0] is sample


@pytest.mark.utils
def test_get_dataset_handler_hf_missing_labels():
    ds = datasets.Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})
    with pytest.raises(ValueError, match="must contain 'labels' key"):
        get_dataset_handler(ds)
