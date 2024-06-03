import pytest

from src.metrics.unnamed.top_k_overlap import TopKOverlap


@pytest.mark.unnamed_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, top_k, batch_size, explanations, expected_score",
    [
        ("mnist", "load_mnist_model", "load_mnist_dataset", 3, 10, "load_mnist_explanations_1", 10),
    ],
)
def test_top_k_overlap_metrics(test_id, model, dataset, top_k, batch_size, explanations, expected_score, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    metric = TopKOverlap(model=model, train_dataset=dataset, top_k=top_k, device="cpu")
    metric.update(explanations=explanations)
    score = metric.compute()
    assert score == expected_score
