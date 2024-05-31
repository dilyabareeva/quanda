import pytest

from metrics.localization.identical_class import IdenticalClass


@pytest.mark.localization_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, test_labels, batch_size, explanations, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_labels_1",
            8,
            "load_mnist_explanations_1",
            0.1,
        ),
    ],
)
def test_identical_class_metrics(
    test_id, model, dataset, test_labels, batch_size, explanations, expected_score, request
):
    model = request.getfixturevalue(model)
    test_labels = request.getfixturevalue(test_labels)
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    metric = IdenticalClass(model=model, train_dataset=dataset, device="cpu")
    metric.update(test_labels=test_labels, explanations=tda)
    score = metric.compute()
    # TODO: introduce a more meaningfull test, where the score is not zero
    assert score == expected_score
