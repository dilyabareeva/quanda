import pytest

from metrics.localization.identical_class import IdenticalClass


@pytest.mark.localization_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, batch_size, explanations",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            8,
            "load_mnist_explanations_1",
        ),
    ],
)
def test_identical_class_metrics(test_id, model, dataset, test_tensor, batch_size, explanations, request):
    model = request.getfixturevalue(model)
    test_tensor = request.getfixturevalue(test_tensor)
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    metric = IdenticalClass(device="cpu")
    score = metric(model=model, train_dataset=dataset, test_dataset=test_tensor, explanations=tda)["score"]
    # TODO: introduce a more meaningfull test, where the score is not zero
    assert score == 0
