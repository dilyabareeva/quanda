import pytest

from src.explainers.wrappers.captum_influence import (
    captum_similarity_self_influence,
)
from src.metrics.unnamed.dataset_cleaning import DatasetCleaning
from src.metrics.unnamed.top_k_overlap import TopKOverlap
from src.utils.functions.similarities import cosine_similarity


@pytest.mark.unnamed_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, top_k, batch_size, explanations, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            3,
            8,
            "load_mnist_explanations_1",
            7,
        ),
    ],
)
def test_top_k_overlap_metrics(
    test_id,
    model,
    dataset,
    top_k,
    batch_size,
    explanations,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    metric = TopKOverlap(model=model, train_dataset=dataset, top_k=top_k, device="cpu")
    metric.update(explanations=explanations)
    score = metric.compute()
    assert score == expected_score


@pytest.mark.unnamed_metrics
@pytest.mark.parametrize(
    "test_id,model,dataset,explanations,global_method,top_k,si_fn,si_fn_kwargs," "batch_size,expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_explanations_1",
            "sum_abs",
            50,
            None,
            None,
            None,
            0.0,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_explanations_1",
            "self-influence",
            50,
            captum_similarity_self_influence,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            8,
            0.0,
        ),
    ],
)
def test_dataset_cleaning(
    test_id,
    model,
    dataset,
    explanations,
    global_method,
    top_k,
    si_fn,
    si_fn_kwargs,
    batch_size,
    expected_score,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)

    if global_method != "self-influence":
        metric = DatasetCleaning(
            model=model,
            train_dataset=dataset,
            global_method=global_method,
            top_k=top_k,
            device="cpu",
        )
        metric.update(explanations=explanations)
    else:
        metric = DatasetCleaning(
            model=model,
            train_dataset=dataset,
            global_method=global_method,
            top_k=top_k,
            si_fn=si_fn,
            si_fn_kwargs=si_fn_kwargs,
            model_id=test_id,
            cache_dir=str(tmp_path),
            batch_size=batch_size,
            device="cpu",
        )
    score = metric.compute()
    assert score == expected_score
