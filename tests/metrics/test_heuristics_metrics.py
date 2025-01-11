import math

import pytest
import torch

from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.heuristics import (
    ModelRandomizationMetric,
    TopKCardinalityMetric,
)
from quanda.metrics.heuristics.mixed_datasets import MixedDatasetsMetric
from quanda.utils.functions import correlation_functions, cosine_similarity
from quanda.utils.common import (
    get_parent_module_from_name,
)


@pytest.mark.heuristic_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, batch_size, explainer_cls, \
    expl_kwargs, explanations, test_labels, correlation_fn",
    [
        (
            "mnist_update_only_spearman",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_labels_1",
            "spearman",
        ),
        (
            "mnist_update_only_kendall",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_labels_1",
            "kendall",
        ),
    ],
)
def test_randomization_metric(
    test_id,
    model,
    checkpoint,
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
    # 1) Check if the metric works correctly
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    test_data = request.getfixturevalue(test_data)
    dataset = request.getfixturevalue(dataset)
    test_labels = request.getfixturevalue(test_labels)
    tda = request.getfixturevalue(explanations)
    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}

    def _load_flexible_state_dict(model: torch.nn.Module, path: str):
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint, strict=False)
        return model

    metric = ModelRandomizationMetric(
        model=model,
        model_id=0,
        checkpoints=checkpoint,
        checkpoints_load_func=_load_flexible_state_dict,
        train_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        cache_dir=str(tmp_path),
        seed=42,
    )
    metric.update(
        test_data=test_data, explanations=tda, explanation_targets=test_labels
    )

    out = metric.compute()["score"]
    assert (out >= -1.0) & (out <= 1.0), "Test failed."

    # 2) Check if the randomization works correctly
    batch_size = 2
    input_shape = test_data[0].shape
    random_tensor = torch.randn((batch_size, *input_shape), device="cpu")

    rand_model = metric._randomize_model()[0]
    rand_model.eval()
    model.eval()

    with torch.no_grad():
        original_out = model(random_tensor)
        randomized_out = rand_model(random_tensor)

    assert not torch.allclose(
        original_out, randomized_out
    ), "Outputs do not differ after randomization"
    assert not torch.isnan(
        randomized_out
    ).any(), "Randomized model output contains NaNs."

    # 3) Check if the randomization works correctly for custom parameters
    model.custom_param = torch.nn.Parameter(torch.randn(4))
    model.eval()
    rand_model = metric._randomize_model()[0]
    rand_model.eval()

    with torch.no_grad():
        original_out = model(random_tensor)
        randomized_out = rand_model(random_tensor)

    assert not torch.allclose(
        original_out, randomized_out
    ), "Outputs do not differ after randomization"
    assert not torch.isnan(
        randomized_out
    ).any(), "Randomized model output contains NaNs."


@pytest.mark.heuristic_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, explainer_cls, expl_kwargs, corr_fn",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
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
            "load_mnist_last_checkpoint",
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
def test_randomization_metric_model_randomization(
    test_id,
    model,
    checkpoint,
    dataset,
    explainer_cls,
    expl_kwargs,
    corr_fn,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}
    metric = ModelRandomizationMetric(
        model=model,
        model_id="0",
        cache_dir=str(tmp_path),
        checkpoints=checkpoint,
        train_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        seed=42,
        correlation_fn=corr_fn,
    )
    rand_model = metric.rand_model
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(), rand_model.named_parameters()
    ):
        parent = get_parent_module_from_name(rand_model, name1)
        if isinstance(parent, (torch.nn.Linear)):
            assert not torch.allclose(param1.data, param2.data), "Test failed."


@pytest.mark.heuristic_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, top_k, batch_size, explanations, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            3,
            8,
            "load_mnist_explanations_similarity_1",
            0.23333333333333334,
        ),
    ],
)
def test_top_k_cardinality_metrics(
    test_id,
    model,
    checkpoint,
    dataset,
    top_k,
    batch_size,
    explanations,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    metric = TopKCardinalityMetric(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
        top_k=top_k,
    )
    metric.update(explanations=explanations)
    score = metric.compute()["score"]
    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, explanations, adversarial_indices, expected_score",
    [
        (
            "mnist_1",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "load_mnist_adversarial_indices",
            0.4699999690055847,
        ),
    ],
)
def test_mixed_datasets_metric(
    test_id,
    model,
    checkpoint,
    dataset,
    explanations,
    adversarial_indices,
    expected_score,
    request,
):
    # Load fixtures using request.getfixturevalue
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    adversarial_indices = request.getfixturevalue(adversarial_indices)

    # Initialize the MixedDatasetsMetric
    metric = MixedDatasetsMetric(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
        adversarial_indices=adversarial_indices,
    )

    # Update the metric with the provided explanations
    metric.update(explanations=explanations)

    # Compute the score
    score = metric.compute()["score"]

    # Validate that the computed score matches the expected score within tolerance
    assert math.isclose(score, expected_score, abs_tol=0.00001)
