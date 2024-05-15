import pytest

from metrics.unnamed.top_k_overlap import TopKOverlap


@pytest.mark.unnamed_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, top_k, batch_size, explanations, expected_score",
    [
        ("mnist", "load_mnist_model", "load_mnist_dataset", 3, 8, "load_mnist_explanations_1", 10),
    ],
)
def test_top_k_overlap_metrics(test_id, model, dataset, top_k, batch_size, explanations, expected_score, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    metric = TopKOverlap(device="cpu")
    score = metric(model=model, train_dataset=dataset, top_k=top_k, explanations=explanations, batch_size=batch_size)[
        "score"
    ]
    assert score == expected_score
