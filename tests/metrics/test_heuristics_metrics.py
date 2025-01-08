import os
import copy
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
    get_load_state_dict_func,
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
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    test_data = request.getfixturevalue(test_data)
    dataset = request.getfixturevalue(dataset)
    test_labels = request.getfixturevalue(test_labels)
    tda = request.getfixturevalue(explanations)
    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}
    metric = ModelRandomizationMetric(
        model=model,
        model_id=0,
        checkpoints=checkpoint,
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


@pytest.mark.heuristic_metrics
@pytest.mark.parametrize(
    "test_id, model, seed",
    [
        ("transformer_randomization", "transformer_model", 42),
    ],
)
def test_randomization_metric_transformer(
    test_id, model, seed, tmp_path, request
):
    model = request.getfixturevalue(model)
    model_id = "transformer"
    cache_dir = str(tmp_path)
    device = "cpu"
    checkpoints = [os.path.join(cache_dir, "dummy.ckpt")]
    checkpoints_load_func = get_load_state_dict_func(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # 1) Check if the parameters are correctly randomized
    original_params = {
        name: p.clone().detach() for name, p in model.named_parameters()
    }

    def _randomize_model(
        model,
        model_id,
        cache_dir,
        checkpoints,
        checkpoints_load_func,
        generator,
    ):
        rand_model = copy.deepcopy(model)
        rand_checkpoints = []

        for i, _ in enumerate(checkpoints):
            for name, param in list(rand_model.named_parameters()):
                parent = get_parent_module_from_name(rand_model, name)
                param_name = name.split(".")[-1]

                if "weight" in name:
                    if isinstance(
                        parent,
                        (
                            torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d,
                        ),
                    ):
                        torch.nn.init.ones_(param)
                    elif (
                        isinstance(
                            parent, (torch.nn.LayerNorm, torch.nn.Embedding)
                        )
                        or param.dim() == 1
                    ):
                        torch.nn.init.normal_(param)
                    else:
                        torch.nn.init.kaiming_normal_(
                            param, generator=generator
                        )
                else:
                    torch.nn.init.normal_(param)

                parent.__setattr__(param_name, torch.nn.Parameter(param))

            chckpt_path = os.path.join(cache_dir, f"{model_id}_rand_{i}.pth")
            torch.save(
                rand_model.state_dict(),
                chckpt_path,
            )
            rand_checkpoints.append(chckpt_path)

        return rand_model, rand_checkpoints

    rand_model, _ = _randomize_model(
        model,
        model_id,
        cache_dir,
        checkpoints,
        checkpoints_load_func,
        generator,
    )

    for (name, original_param), (_, rand_param) in zip(
        original_params.items(), rand_model.named_parameters()
    ):
        assert not torch.allclose(
            original_param, rand_param
        ), f"Parameter {name} was not randomized."
        assert not torch.isnan(
            rand_param
        ).any(), f"Parameter {name} contains NaN values."

    # 2) Check if the output of the randomized model is not NaN
    rand_model.eval()
    batch_size = 2
    seq_len = 10
    vocab_size = 100
    x = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        out = rand_model(x)

    assert not torch.isnan(out).any(), "Randomized model output contains NaNs."


@pytest.mark.heuristic_metrics
@pytest.mark.parametrize(
    "test_id, model, seed",
    [
        ("batchnorm_randomization", "batchnorm_model", 42),
    ],
)
def test_randomization_metric_batchnorm(
    test_id, model, seed, tmp_path, request
):
    model = request.getfixturevalue(model)
    model_id = "batchnorm"
    cache_dir = str(tmp_path)
    device = "cpu"
    checkpoint_path = os.path.join(tmp_path, "dummy.ckpt")
    torch.save(model.state_dict(), checkpoint_path)
    checkpoints = [checkpoint_path]
    checkpoints_load_func = get_load_state_dict_func(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    def _randomize_model(
        model,
        model_id,
        cache_dir,
        checkpoints,
        checkpoints_load_func,
        generator,
    ):
        rand_model = copy.deepcopy(model)
        rand_checkpoints = []

        for i, chckpt in enumerate(checkpoints):
            checkpoints_load_func(rand_model, chckpt)

            for name, param in list(rand_model.named_parameters()):
                parent = get_parent_module_from_name(rand_model, name)
                param_name = name.split(".")[-1]

                if "weight" in name:
                    if isinstance(
                        parent,
                        (
                            torch.nn.BatchNorm1d,
                            torch.nn.BatchNorm2d,
                            torch.nn.BatchNorm3d,
                        ),
                    ):
                        torch.nn.init.ones_(param)
                    elif (
                        isinstance(
                            parent, (torch.nn.LayerNorm, torch.nn.Embedding)
                        )
                        or param.dim() == 1
                    ):
                        torch.nn.init.normal_(param)
                    else:
                        torch.nn.init.kaiming_normal_(
                            param, generator=generator
                        )
                else:
                    torch.nn.init.normal_(param)

                parent.__setattr__(param_name, torch.nn.Parameter(param))

            chckpt_path = os.path.join(cache_dir, f"{model_id}_rand_{i}.pth")
            torch.save(rand_model.state_dict(), chckpt_path)
            rand_checkpoints.append(chckpt_path)

        return rand_model, rand_checkpoints

    rand_model, _ = _randomize_model(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        checkpoints=checkpoints,
        checkpoints_load_func=checkpoints_load_func,
        generator=generator,
    )

    rand_model.eval()
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28, device=device)

    out = None
    with torch.no_grad():
        out = rand_model(x)

    assert not torch.isnan(
        out
    ).any(), "Output contains NaNs after randomization!"


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
