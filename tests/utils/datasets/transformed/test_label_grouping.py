import pytest

from quanda.utils.datasets.transformed import (
    ClassMapping,
    LabelGroupingDataset,
)


@pytest.mark.utils
@pytest.mark.parametrize(
    "dataset, n_classes, n_groups, seed, expected_score",
    [
        (
            "load_mnist_dataset",
            10,
            2,
            27,
            0.375,
        ),
    ],
)
def test_label_grouping_dataset(
    dataset,
    n_classes,
    n_groups,
    seed,
    expected_score,
    request,
):
    dataset = request.getfixturevalue(dataset)

    metadata = LabelGroupingDataset.metadata_cls(
        cls_idx=0,
        p=0.5,
    )
    mapping = ClassMapping(
        class_to_group=ClassMapping._generate(n_classes, n_groups, seed),
        n_classes=n_classes,
        n_groups=n_groups,
        seed=seed,
    )
    grouped_dataset = LabelGroupingDataset(
        dataset=dataset,
        metadata=metadata,
        class_to_group=mapping.class_to_group,
        n_classes=mapping.n_classes,
        n_groups=mapping.n_groups,
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
