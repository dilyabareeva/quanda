import pytest
import torch

from src.explainers.aggregators import AbsSumAggregator, SumAggregator


@pytest.mark.aggregators
@pytest.mark.parametrize(
    "test_id, explanations",
    [
        (
            "mnist",
            "load_mnist_explanations_1",
        ),
    ],
)
def test_sum_aggregator(test_id, explanations, request):
    explanations = request.getfixturevalue(explanations)
    aggregator = SumAggregator()
    aggregator.update(explanations)
    global_rank = aggregator.compute().argsort()
    assert torch.allclose(global_rank, explanations.sum(dim=0).argsort())


@pytest.mark.aggregators
@pytest.mark.parametrize(
    "test_id, explanations",
    [
        (
            "mnist",
            "load_mnist_explanations_1",
        ),
    ],
)
def test_abs_aggregator(test_id, explanations, request):
    explanations = request.getfixturevalue(explanations)
    aggregator = AbsSumAggregator()
    aggregator.update(explanations)
    global_rank = aggregator.compute().argsort()
    assert torch.allclose(global_rank, explanations.abs().mean(dim=0).argsort())
