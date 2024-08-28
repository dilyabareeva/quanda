import pytest
import torch

from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.tasks.global_ranking import GlobalRanking
from quanda.utils.functions.similarities import cosine_similarity


@pytest.mark.toy_benchmarks
@pytest.mark.parametrize(
    "test_id, model, dataset, " "global_method, batch_size, explainer_cls, expl_kwargs, ",
    [
        (
            "mnist1",
            "load_mnist_model",
            "load_mnist_dataset",
            "self-influence",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
        ),
    ],
)
def test_global_ranking(
    test_id,
    model,
    dataset,
    global_method,
    batch_size,
    explainer_cls,
    expl_kwargs,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    expl_kwargs = {**expl_kwargs, "model_id": "test", "cache_dir": str(tmp_path)}

    global_ranker = GlobalRanking(
        model=model,
        train_dataset=dataset,
        global_method=global_method,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        device="cpu",
    )
    ranking = global_ranker.compute()

    assert len(ranking) == len(dataset)


@pytest.mark.toy_benchmarks
@pytest.mark.parametrize(
    "test_id, model, dataset, " "global_method, batch_size, explainer_cls, expl_kwargs,",
    [
        (
            "mnist1",
            "load_mnist_model",
            "load_mnist_dataset",
            "self-influence",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
        ),
    ],
)
def test_global_ranking_loading_saving_resetting(
    test_id,
    model,
    dataset,
    global_method,
    batch_size,
    explainer_cls,
    expl_kwargs,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    expl_kwargs = {**expl_kwargs, "model_id": "test", "cache_dir": str(tmp_path)}

    global_ranker = GlobalRanking(
        model=model,
        train_dataset=dataset,
        global_method=global_method,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        device="cpu",
    )
    ranking_1 = global_ranker.compute()
    state_dict = global_ranker.state_dict()
    global_ranker.reset()
    global_ranker.load_state_dict(state_dict)
    ranking_2 = global_ranker.compute()

    assert torch.allclose(ranking_1, ranking_2)
