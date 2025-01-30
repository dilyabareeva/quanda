import math

import pytest
import torch
from torch.nn.modules.batchnorm import _BatchNorm

from quanda.explainers.wrappers import CaptumSimilarity, CaptumTracInCP
from quanda.metrics.heuristics import (
    ModelRandomizationMetric,
    TopKCardinalityMetric,
)
from quanda.metrics.heuristics.mixed_datasets import MixedDatasetsMetric
from quanda.utils.functions import cosine_similarity
from quanda.utils.common import (
    get_parent_module_from_name,
)


@pytest.mark.heuristic_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint, dataset, test_data, "
    "explainer_cls, expl_kwargs, explanations, test_labels",
    [
        (
            "randomization_metric",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_labels_1",
        )
    ],
)
def test_randomization_metric_score(
    test_id,
    model,
    checkpoint,
    dataset,
    test_data,
    explainer_cls,
    expl_kwargs,
    explanations,
    test_labels,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    test_data = request.getfixturevalue(test_data)
    dataset = request.getfixturevalue(dataset)
    test_labels = request.getfixturevalue(test_labels)
    tda = request.getfixturevalue(explanations)
    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}

    metric = ModelRandomizationMetric(
        model=model,
        model_id="0",
        checkpoints=checkpoint,
        train_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        cache_dir=str(tmp_path),
        seed=42,
    )
    metric.update(
        test_data=test_data, explanations=tda, test_targets=test_labels
    )

    out = metric.compute()["score"]
    assert (out >= -1.0) & (
        out <= 1.0
    ), "Metric score is out of expected range."


@pytest.mark.heuristic_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint, checkpoints_load_func, dataset, input_shape, "
    "test_data, batch_size, explainer_cls",
    [
        (
            "randomization_lenet",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            None,
            "load_mnist_dataset",
            (1, 28, 28),
            "load_mnist_test_samples_1",
            8,
            CaptumTracInCP,
        ),
        (
            "randomization_vit",
            "load_vit",
            "load_mnist_last_checkpoint",
            lambda x: x,
            "load_mnist_dataset",
            (3, 224, 224),
            "load_mnist_test_samples_1",
            8,
            CaptumTracInCP,
        ),
        (
            "randomization_resnet",
            "load_resnet",
            "load_mnist_last_checkpoint",
            lambda x: x,
            "load_mnist_dataset",
            (3, 224, 224),
            "load_mnist_test_samples_1",
            8,
            CaptumTracInCP,
        ),
        (
            "randomization_custom_param",
            "load_mnist_model_with_custom_param",
            "load_mnist_last_checkpoint",
            lambda x: x,
            "load_mnist_dataset",
            (1, 28, 28),
            "load_mnist_test_samples_1",
            8,
            CaptumTracInCP,
        ),
    ],
)
def test_randomization_metric_output_nan(
    test_id,
    model,
    checkpoint,
    checkpoints_load_func,
    dataset,
    input_shape,
    test_data,
    batch_size,
    explainer_cls,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    if checkpoint is not None:
        checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)

    metric = ModelRandomizationMetric(
        model=model,
        model_id="0",
        checkpoints=checkpoint,
        checkpoints_load_func=lambda x, y: x,
        train_dataset=dataset,
        explainer_cls=explainer_cls,
        cache_dir=str(tmp_path),
        seed=42,
    )

    # Generate a random batch of data
    random_tensor = torch.randn((batch_size, *input_shape), device="cpu")

    # Randomize model
    rand_model = metric._randomize_model()[0]
    rand_model.eval()

    # Check if the outputs differ after randomization
    with torch.no_grad():
        randomized_out = rand_model(random_tensor)

    assert not torch.isnan(
        randomized_out
    ).any(), "Randomized model output contains NaNs."


@pytest.mark.heuristic_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint, checkpoints_load_func, dataset, input_shape, "
    "test_data, batch_size, explainer_cls",
    [
        (
            "randomization_lenet",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            None,
            "load_mnist_dataset",
            (1, 28, 28),
            "load_mnist_test_samples_1",
            8,
            CaptumTracInCP,
        ),
        (
            "randomization_vit",
            "load_vit",
            "load_mnist_last_checkpoint",
            lambda x: x,
            "load_mnist_dataset",
            (3, 224, 224),
            "load_mnist_test_samples_1",
            8,
            CaptumTracInCP,
        ),
        (
            "randomization_resnet",
            "load_resnet",
            "load_mnist_last_checkpoint",
            lambda x: x,
            "load_mnist_dataset",
            (3, 224, 224),
            "load_mnist_test_samples_1",
            8,
            CaptumTracInCP,
        ),
        (
            "randomization_custom",
            "load_mnist_model_with_custom_param",
            "load_mnist_last_checkpoint",
            lambda x: x,
            "load_mnist_dataset",
            (1, 28, 28),
            "load_mnist_test_samples_1",
            8,
            CaptumTracInCP,
        ),
    ],
)
def test_randomization_metric_randomization(
    test_id,
    model,
    checkpoint,
    checkpoints_load_func,
    dataset,
    input_shape,
    test_data,
    batch_size,
    explainer_cls,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    if checkpoint is not None:
        checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)

    metric = ModelRandomizationMetric(
        model=model,
        model_id="0",
        checkpoints=checkpoint,
        checkpoints_load_func=lambda x, y: x,
        train_dataset=dataset,
        explainer_cls=explainer_cls,
        cache_dir=str(tmp_path),
        seed=42,
    )

    rand_model = metric.rand_model
    for (name1, param1), (name2, param2) in zip(
        model.named_parameters(), rand_model.named_parameters()
    ):
        parent = get_parent_module_from_name(rand_model, name1)
        if (not isinstance(parent, (_BatchNorm))) and (
            not isinstance(parent, torch.nn.LayerNorm)
        ):
            assert not torch.allclose(
                param1.data, param2.data
            ), "Randomized model output contains NaNs."


@pytest.mark.heuristic_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, top_k, batch_size, explanations, expected_score",
    [
        (
            "ylecun/mnist",
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
