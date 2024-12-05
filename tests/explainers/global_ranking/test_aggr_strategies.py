import pytest
import torch

from quanda.explainers import AbsSumAggregator
from quanda.explainers.global_ranking import (
    GlobalAggrStrategy,
    GlobalSelfInfluenceStrategy,
)
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.functions import cosine_similarity


@pytest.mark.aggr_strategies
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, explainer_cls, expl_kwargs, expected",
    [
        (
            "mnist_si",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
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
    checkpoint,
    dataset,
    test_data,
    explainer_cls,
    expl_kwargs,
    expected,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    test_data = request.getfixturevalue(test_data)
    dataset = request.getfixturevalue(dataset)

    explainer = explainer_cls(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
        cache_dir=str(tmp_path),
        model_id="mnist_model",
        **expl_kwargs,
    )
    aggr_strat = GlobalSelfInfluenceStrategy(explainer=explainer)
    si = aggr_strat.get_self_influence()
    assert torch.allclose(si, expected)


@pytest.mark.aggr_strategies
@pytest.mark.parametrize(
    "test_id, explanations, expected",
    [
        (
            "mnist_aggr",
            "mnist_range_explanations",
            "range_ranking",
        ),
    ],
)
def test_global_aggr_strategy(
    test_id,
    explanations,
    expected,
    request,
):
    explanations = request.getfixturevalue(explanations)
    expected = request.getfixturevalue(expected)

    aggr_strat = GlobalAggrStrategy(aggr_type=AbsSumAggregator)
    aggr_strat.update(explanations=explanations)
    assert torch.allclose(aggr_strat.get_global_rank(), expected)
