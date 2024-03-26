import pytest

from src.utils.datasets.corrupt_label_dataset import CorruptLabelDataset


@pytest.mark.utils
def test_corrupt_label_dataset(dataset):
    cl_dataset = CorruptLabelDataset(dataset)
    assert len(cl_dataset[0][1]) == 2
