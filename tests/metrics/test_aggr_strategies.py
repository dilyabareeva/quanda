import pytest
import torch

from src.explainers.aggregators import AbsSumAggregator
from src.explainers.wrappers.captum_influence import CaptumSimilarity
from src.metrics.aggr_strategies import (
    GlobalAggrStrategy,
    GlobalSelfInfluenceStrategy,
)
from src.utils.functions.similarities import cosine_similarity


@pytest.mark.aggr_strategies
@pytest.mark.parametrize(
    "test_id, model, dataset, test_data, explainer_cls, expl_kwargs, expected",
    [
        (
            "mnist_si",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            torch.ones((8,)),
        ),
    ],
)
def test_global_self_influence_strategy(
    test_id,
    model,
    dataset,
    test_data,
    explainer_cls,
    expl_kwargs,
    expected,
    request,
):
    model = request.getfixturevalue(model)
    test_data = request.getfixturevalue(test_data)
    dataset = request.getfixturevalue(dataset)

    explainer = explainer_cls(
        model=model, train_dataset=dataset, cache_dir="cache_dir_aggr_strat_test", model_id="mnist_model", **expl_kwargs
    )
    aggr_strat = GlobalSelfInfluenceStrategy(explainer=explainer)
    si = aggr_strat.get_self_influence()
    assert torch.allclose(si, expected)


@pytest.mark.aggr_strategies
@pytest.mark.parametrize(
    "test_id, model, dataset, test_data, explanations, expected",
    [
        (
            "mnist_aggr",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "mnist_range_explanations",
            "range_ranking",
        ),
    ],
)
def test_global_aggr_strategy(
    test_id,
    model,
    dataset,
    test_data,
    explanations,
    expected,
    request,
):
    model = request.getfixturevalue(model)
    test_data = request.getfixturevalue(test_data)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    expected = request.getfixturevalue(expected)

    aggr_strat = GlobalAggrStrategy(aggr_type=AbsSumAggregator)
    aggr_strat.update(explanations=explanations)
    assert torch.allclose(aggr_strat.get_global_rank(), expected)