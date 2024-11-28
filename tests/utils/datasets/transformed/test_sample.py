import pytest
import torch

from quanda.utils.datasets.transformed import SampleTransformationDataset


@pytest.mark.utils
@pytest.mark.parametrize(
    "dataset, n_classes, seed, transform_indices, transformation, err",
    [
        (
            "load_mnist_dataset",
            10,
            27,
            [0, 1, 2, 3],
            "mnist_white_square_transformation",
            None,
        ),
        (
            "load_mnist_dataset",
            10,
            27,
            None,
            "mnist_white_square_transformation",
            None,
        ),
    ],
)
def test_sample_transformation_dataset(
    dataset, n_classes, seed, transform_indices, transformation, err, request
):
    dataset = request.getfixturevalue(dataset)
    transformation = request.getfixturevalue(transformation)

    sample_dataset = SampleTransformationDataset(
        dataset=dataset,
        n_classes=n_classes,
        seed=seed,
        transform_indices=transform_indices,
        sample_fn=transformation,
    )

    if transform_indices is not None:
        all_equal = [
            torch.allclose(sample_dataset[i][0], transformation(dataset[i][0]))
            for i in transform_indices
        ]
        assert all(all_equal)

    else:
        all_equal = [
            torch.allclose(sample_dataset[i][0], transformation(dataset[i][0]))
            for i in sample_dataset.transform_indices
        ]
        assert all(all_equal)
