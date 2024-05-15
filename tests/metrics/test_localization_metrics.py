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
    metric = IdenticalClass(device="cpu")
    score = metric(model=model, train_dataset=dataset, test_labels=test_labels, explanations=tda)["score"]
    # TODO: introduce a more meaningfull test, where the score is not zero
    # Note from Galip:
    # one idea could be: a random attributor should get approximately 1/( # of classes).
    # With a big test dataset, the probability of failing a truly random test
    # should diminish.
    assert score == expected_score