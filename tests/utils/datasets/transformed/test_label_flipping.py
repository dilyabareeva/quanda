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
                n_classes=n_classes,
                metadata=metadata,
            )
        return
    else:
        flipped_dataset = LabelFlippingDataset(
            dataset=dataset,
            n_classes=n_classes,
            metadata=metadata,
        )
        assertions = []
        labels = flipped_dataset.mislabeling_labels
        assertions.append(len(labels.keys()) == len(expected.keys()))
        for i in labels.keys():
            assertions.append(labels[i] == expected[i])
        assert all(assertions)
