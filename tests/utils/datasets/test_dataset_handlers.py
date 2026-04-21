import datasets
import pytest
import torch

from quanda.utils.datasets.dataset_handlers import (
    HuggingFaceDatasetHandler,
    HuggingFaceSequenceDatasetHandler,
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


@pytest.mark.utils
def test_huggingface_sequence_handler_emits_ordered_lists():
    handler = HuggingFaceSequenceDatasetHandler(
        input_keys=("input_ids", "attention_mask"),
    )
    ds = datasets.Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5, 6]],
            "attention_mask": [[1, 1, 1], [1, 1, 0]],
            "labels": [0, 1],
        }
    )
    ds.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    loader = handler.create_dataloader(ds, batch_size=2)
    batch = next(iter(loader))

    assert isinstance(batch, list)
    assert len(batch) == 3
    input_ids, attention_mask, labels = batch
    assert torch.equal(input_ids, torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert torch.equal(attention_mask, torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert torch.equal(labels, torch.tensor([0, 1]))


@pytest.mark.utils
def test_huggingface_sequence_handler_process_batch_returns_dict():
    handler = HuggingFaceSequenceDatasetHandler(
        input_keys=("input_ids", "attention_mask"),
    )
    batch = [
        torch.tensor([[1, 2]]),
        torch.tensor([[1, 1]]),
        torch.tensor([0]),
    ]
    inputs, labels = handler.process_batch(batch, device="cpu")

    assert set(inputs.keys()) == {"input_ids", "attention_mask"}
    assert torch.equal(inputs["input_ids"], batch[0])
    assert torch.equal(inputs["attention_mask"], batch[1])
    assert torch.equal(labels, batch[2])


@pytest.mark.utils
def test_torch_handler_build_positional_batch():
    handler = TorchDatasetHandler()
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    labels = torch.tensor([0, 1])

    out_inputs, out_labels = handler.build_positional_batch(
        inputs, labels, device="cpu"
    )

    assert torch.equal(out_inputs, inputs)
    assert torch.equal(out_labels, labels)


@pytest.mark.utils
def test_huggingface_sequence_handler_build_positional_batch():
    handler = HuggingFaceSequenceDatasetHandler(
        input_keys=("input_ids", "attention_mask"),
    )
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    labels = torch.tensor([0])

    batch = handler.build_positional_batch(inputs, labels, device="cpu")

    assert len(batch) == 3
    assert torch.equal(batch[0], inputs["input_ids"])
    assert torch.equal(batch[1], inputs["attention_mask"])
    assert torch.equal(batch[2], labels)


@pytest.mark.utils
def test_get_dataset_handler_unwraps_subset_and_raises_on_unsupported():
    tensor_ds = torch.utils.data.TensorDataset(
        torch.randn(4, 2), torch.randint(0, 2, (4,))
    )
    subset = torch.utils.data.Subset(tensor_ds, [0, 1])
    assert isinstance(get_dataset_handler(subset), TorchDatasetHandler)

    with pytest.raises(ValueError, match="Unsupported dataset type"):
        get_dataset_handler("not a dataset")  # type: ignore[arg-type]
