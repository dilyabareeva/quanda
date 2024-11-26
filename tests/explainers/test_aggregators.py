import pytest
import torch

from quanda.explainers import AbsSumAggregator, SumAggregator


@pytest.mark.aggregators
@pytest.mark.parametrize(
    "test_id, explanations, aggregator, expected",
    [
        (
            "mnist0",
            "load_mnist_explanations_similarity_1",
            AbsSumAggregator,
            {},
        ),
        ("mnist1", "load_mnist_explanations_similarity_1", SumAggregator, {}),
        (
            "mnist2",
            "load_mnist_explanations_similarity_1",
            SumAggregator,
            {"err_expl": ValueError},
        ),
        (
            "mnist3",
            "load_mnist_explanations_similarity_1",
            SumAggregator,
            {"err_reset": ValueError},
        ),
    ],
)
def test_aggregator_update(
    test_id, explanations, aggregator, expected, request
):
    explanations = request.getfixturevalue(explanations)
    aggregator = aggregator()
    aggregator.update(explanations)
    if "err_expl" in expected:
        explanations = torch.randn((10, 10))
        with pytest.raises(expected["err_expl"]):
            aggregator.update(explanations)
    else:
        aggregator.update(explanations)
        if "err_reset" in expected:
            aggregator.reset()
            with pytest.raises(expected["err_reset"]):
                global_rank = aggregator.compute().argsort()
        else:
            global_rank = aggregator.compute().argsort()
            assert torch.allclose(
                global_rank, explanations.sum(dim=0).argsort()
            )


@pytest.mark.aggregators
@pytest.mark.parametrize(
    "test_id, explanations, aggregator",
    [
        ("mnist", "load_mnist_explanations_similarity_1", AbsSumAggregator),
        ("mnist", "load_mnist_explanations_similarity_1", SumAggregator),
    ],
)
def test_aggregator_reset(test_id, explanations, aggregator, request):
    explanations = request.getfixturevalue(explanations)
    aggregator = aggregator()
    aggregator.update(explanations=explanations)
    aggregator.reset()
    assert aggregator.scores is None


@pytest.mark.aggregators
@pytest.mark.parametrize(
    "test_id, explanations, aggregator",
    [
        ("mnist", "load_mnist_explanations_similarity_1", AbsSumAggregator),
        ("mnist", "load_mnist_explanations_similarity_1", SumAggregator),
    ],
)
def test_aggregator_save(test_id, explanations, aggregator, request):
    explanations = request.getfixturevalue(explanations)
    aggregator = aggregator()
    aggregator.update(explanations=explanations)
    sdict = aggregator.state_dict
    assert torch.allclose(sdict["scores"], aggregator.scores)


@pytest.mark.aggregators
@pytest.mark.parametrize(
    "test_id, explanations, aggregator",
    [
        ("mnist", "load_mnist_explanations_similarity_1", AbsSumAggregator),
        ("mnist", "load_mnist_explanations_similarity_1", SumAggregator),
    ],
)
def test_aggregator_load(test_id, explanations, aggregator, request):
    explanations = request.getfixturevalue(explanations)
    aggregator = aggregator()
    aggregator.update(explanations=explanations)
    sdict = aggregator.state_dict
    aggregator.reset()
    aggregator.load_state_dict(sdict)
    assert torch.allclose(sdict["scores"], aggregator.scores)
