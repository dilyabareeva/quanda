import pytest

from quanda.utils.datasets.transformed import LabelFlippingDataset


@pytest.mark.utils
@pytest.mark.parametrize(
    "dataset, n_classes, seed, flipped_labels, expected, err",
    [
        (
            "load_mnist_dataset",
            10,
            27,
            "mnist_seed_27_mislabeling_labels",
            "mnist_seed_27_mislabeling_labels",
            None,
        ),
        (
            "load_mnist_dataset",
            10,
            27,
            None,
            "mnist_seed_27_mislabeling_labels",
            None,
        ),
        ("load_mnist_dataset", 10, 27, [], None, ValueError),
        ("load_mnist_dataset", 10, 27, {10022: 32, 892: 33}, None, ValueError),
    ],
)
def test_label_flipping_dataset(
    dataset,
    n_classes,
    seed,
    flipped_labels,
    expected,
    err,
    request,
):
    dataset = request.getfixturevalue(dataset)
    if expected is not None:
        expected = request.getfixturevalue(expected)

    metadata = LabelFlippingDataset.metadata_cls(
        seed=seed,
    )
    metadata.transform_indices = metadata.generate_indices(dataset)

    if isinstance(flipped_labels, str):
        flipped_labels = request.getfixturevalue(flipped_labels)

    if flipped_labels is None:
        flipped_labels = metadata.generate_mislabeling_labels(dataset)

    metadata.mislabeling_labels = flipped_labels

    if err is not None:
        with pytest.raises(err):
            flipped_dataset = LabelFlippingDataset(
                dataset=dataset,
                metadata=metadata,
            )
        return
    else:
        flipped_dataset = LabelFlippingDataset(
            dataset=dataset,
            metadata=metadata,
        )
        assertions = []
        labels = flipped_dataset.mislabeling_labels
        assertions.append(len(labels.keys()) == len(expected.keys()))
        for i in labels.keys():
            assertions.append(labels[i] == expected[str(i)])
        assert all(assertions)


@pytest.mark.utils
def test_label_flipping_dataset_hf(load_text_dataset):
    """Covers the HuggingFaceDatasetHandler branch in __getitem__."""
    ds_train, _ = load_text_dataset

    metadata = LabelFlippingDataset.metadata_cls(
        n_classes=2,
        seed=27,
    )
    metadata.transform_indices = [0, 1]
    metadata.mislabeling_labels = {0: 1, 1: 0}

    flipped_dataset = LabelFlippingDataset(
        dataset=ds_train,
        metadata=metadata,
    )

    item0 = flipped_dataset[0]
    item1 = flipped_dataset[1]
    item_untouched = flipped_dataset[2]

    assert isinstance(item0, dict)
    assert item0["labels"] == 1
    assert item1["labels"] == 0
    assert item_untouched["labels"] == ds_train[2]["labels"]
