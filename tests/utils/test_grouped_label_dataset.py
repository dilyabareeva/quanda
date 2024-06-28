import pytest

from src.utils.datasets.transformed.label_grouping import GroupLabelDataset


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
def test_identical_subclass_metrics(
    dataset,
    n_classes,
    n_groups,
    class_to_group,
    seed,
    expected_score,
    request,
):
    dataset = request.getfixturevalue(dataset)

    grouped_dataset = GroupLabelDataset(
        dataset=dataset,
        n_classes=n_classes,
        n_groups=n_groups,
        class_to_group=class_to_group,
        seed=seed,
    )

    assertions = []

    for i in range(len(grouped_dataset)):
        x, g = grouped_dataset[i]
        y = grouped_dataset._get_original_label(i)
        assertions.append((g in range(n_groups)) & (g == grouped_dataset.class_to_group[y.item()]))

    assert all(assertions)
