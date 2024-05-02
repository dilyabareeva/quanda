import pytest

from metrics.localization.identical_class import IdenticalClass


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_prediction, explanations",
    [
        ("load_rand_test_predictions", "load_rand_tensor_explanations"),
    ],
)
def test_identical_class_metrics(test_prediction, explanations, request):
    test_prediction = request.getfixturevalue(test_prediction)
    explanations = request.getfixturevalue(explanations)
    metric = IdenticalClass(device="cpu")
    score = metric(test_prediction, explanations)["score"]
    assert score > 0
