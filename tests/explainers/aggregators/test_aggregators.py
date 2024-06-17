import pytest
import torch

from src.explainers.aggregators.aggregators import (
    AbsSumAggregator,
    SumAggregator,
)


@pytest.mark.aggregators
@pytest.mark.parametrize(
    "test_id, dataset, explanations",
    [
        (
            "mnist",
            "load_mnist_dataset",
            "load_mnist_explanations_1",
        ),
    ],
)
def test_sum_aggregator(test_id, dataset, explanations, request):
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    aggregator = SumAggregator(training_size=len(dataset))
    aggregator.update(explanations)
    global_rank = aggregator.compute()
    assert torch.allclose(global_rank, explanations.sum(dim=0).argsort())


@pytest.mark.aggregators
@pytest.mark.parametrize(
    "test_id, dataset, explanations",
    [
        (
            "mnist",
            "load_mnist_dataset",
            "load_mnist_explanations_1",
        ),
    ],
)
def test_abs_aggregator(test_id, dataset, explanations, request):
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    aggregator = AbsSumAggregator(training_size=len(dataset))
    aggregator.update(explanations)
    global_rank = aggregator.compute()
    assert torch.allclose(global_rank, explanations.abs().mean(dim=0).argsort())
