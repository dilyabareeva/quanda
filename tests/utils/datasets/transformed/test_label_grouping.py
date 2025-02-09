import pytest

from quanda.utils.datasets.transformed import LabelGroupingDataset


@pytest.mark.utils
@pytest.mark.parametrize(
    "dataset, n_classes, n_groups, class_to_group, seed, expected_score",
    [
        (
            "load_mnist_dataset",
            10,
            2,
            "random",
            27,
            0.375,
        ),
    ],
)
def test_label_grouping_dataset(
    dataset,
    n_classes,
    n_groups,
    class_to_group,
    seed,
    expected_score,
    request,
):
    dataset = request.getfixturevalue(dataset)

    metadata = LabelGroupingDataset.metadata_cls(
        cls_idx=0,
        p=0.5,
    )
    grouped_dataset = LabelGroupingDataset(
        dataset=dataset,
        metadata=metadata,
    )

    assertions = []

    for i in range(len(grouped_dataset)):
        x, g = grouped_dataset[i]
        y = grouped_dataset.get_original_label(i)
        assertions.append(
            (i not in grouped_dataset.transform_indices)
            or (g == grouped_dataset.class_to_group[y])
        )

    assert all(assertions)
