import pytest
import torch

from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.metrics.randomization.model_randomization import (
    ModelRandomizationMetric,
)
from quanda.utils.functions.correlations import correlation_functions
from quanda.utils.functions.similarities import cosine_similarity


@pytest.mark.randomization_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, test_data, batch_size, explainer_cls, \
    expl_kwargs, explanations, test_labels, correlation_fn, tmp_path",
    [
        (
            "mnist_update_only_spearman",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            "load_mnist_explanations_1",
            "load_mnist_test_labels_1",
            "spearman",
            "tmp_path",
        ),
        (
            "mnist_update_only_kendall",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            "load_mnist_explanations_1",
            "load_mnist_test_labels_1",
            "kendall",
            "tmp_path",
        ),
        (
            "mnist_explain_update",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            "load_mnist_explanations_1",
            "load_mnist_test_labels_1",
            "spearman",
            "tmp_path",
        ),
    ],
)
def test_randomization_metric(
    test_id,
    model,
    dataset,
    test_data,
    batch_size,
    explainer_cls,
    expl_kwargs,
    explanations,
    test_labels,
    correlation_fn,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    test_data = request.getfixturevalue(test_data)
    dataset = request.getfixturevalue(dataset)
    test_labels = request.getfixturevalue(test_labels)
    tda = request.getfixturevalue(explanations)
    metric = ModelRandomizationMetric(
        model=model,
        train_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        correlation_fn=correlation_fn,
        cache_dir=str(tmp_path),
        seed=42,
        device="cpu",
    )
    # TODO: introduce a more meaningful test
    # Can we come up with a special attributor that gets exactly 0 score?
    if "explain" in test_id:
        metric.explain_update(test_data=test_data, explanation_targets=test_labels)
    else:
        metric.update(test_data=test_data, explanations=tda, explanation_targets=test_labels)

    out = metric.compute()
    assert (out >= -1.0) & (out <= 1.0), "Test failed."


@pytest.mark.randomization_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, explainer_cls, expl_kwargs, corr_fn",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            "spearman",
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            correlation_functions["kendall"],
        ),
    ],
)
def test_randomization_metric_model_randomization(test_id, model, dataset, explainer_cls, expl_kwargs, corr_fn, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    metric = ModelRandomizationMetric(
        model=model,
        train_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        seed=42,
        device="cpu",
        correlation_fn=corr_fn,
    )
    rand_model = metric.rand_model
    for param1, param2 in zip(model.parameters(), rand_model.parameters()):
        assert not torch.allclose(param1.data, param2.data), "Test failed."
