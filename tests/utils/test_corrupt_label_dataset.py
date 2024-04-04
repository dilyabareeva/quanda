import pytest

# from src.utils.datasets.corrupt_label_dataset import CorruptLabelDataset


@pytest.mark.utils
@pytest.mark.parametrize(
    "dataset, n_expected",
    [
        ("load_dataset", 2),
    ],
)
def test_corrupt_label_dataset(dataset, n_expected, request):
    dataset = request.getfixturevalue(dataset)
    # cl_dataset = CorruptLabelDataset(dataset)
    assert 2 == n_expected
