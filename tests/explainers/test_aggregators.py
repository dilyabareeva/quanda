import pytest
import torch

from src.explainers.aggregators import AbsSumAggregator, SumAggregator


@pytest.mark.aggregators
@pytest.mark.parametrize(
    "test_id, explanations, batch_size",
    [
        (
            "mnist",
            "load_mnist_explanations_1",
            1,
        ),
    ],
)
def test_sum_aggregator(test_id, explanations, batch_size, request):
    explanations = request.getfixturevalue(explanations)
    aggregator = SumAggregator()
    for i in range(0, explanations.shape[0], batch_size):
        upper_index = min(explanations.shape[0], i + batch_size)
        aggregator.update(explanations[i:upper_index])
    global_rank = aggregator.compute().argsort()
    assert torch.allclose(global_rank, explanations.sum(dim=0).argsort())


@pytest.mark.aggregators
@pytest.mark.parametrize(
    "test_id, explanations, batch_size",
    [
        (
            "mnist",
            "load_mnist_explanations_1",
            1,
        ),
    ],
)
def test_abs_aggregator(test_id, explanations, batch_size, request):
    explanations = request.getfixturevalue(explanations)
    aggregator = AbsSumAggregator()
    for i in range(0, explanations.shape[0], batch_size):
        upper_index = min(explanations.shape[0], i + batch_size)
        aggregator.update(explanations[i:upper_index])
    global_rank = aggregator.compute().argsort()
    assert torch.allclose(global_rank, explanations.abs().mean(dim=0).argsort())
