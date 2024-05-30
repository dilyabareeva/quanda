import pytest
import torch

from metrics.randomization.model_randomization import ModelRandomizationMetric
from utils.explain_wrapper import explain


@pytest.mark.randomization_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, test_data, batch_size, explain_kwargs, explanations, corr_measure",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            8,
            {"method": "SimilarityInfluence", "layer": "fc_2"},
            "load_mnist_explanations_1",
            "spearman",
        ),
    ],
)
def test_randomization_metric(
    test_id, model, dataset, test_data, batch_size, explain_kwargs, explanations, corr_measure, request
):
    model = request.getfixturevalue(model)
    test_data = request.getfixturevalue(test_data)
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    metric = ModelRandomizationMetric(
        model=model,
        train_dataset=dataset,
        explain_fn=explain,
        explain_fn_kwargs={**explain_kwargs, "layer": "fc_2"},
        correlation_fn="spearman",
        seed=42,
        device="cpu",
    )
    metric.update(test_data, tda)
    out = metric.compute()
    assert (out.item() >= -1.0) and (out.item() <= 1.0), "Test failed."
    assert isinstance(out, torch.Tensor), "Output is not a tensor."


@pytest.mark.randomization_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset,",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
        ),
    ],
)
def test_model_randomization(test_id, model, dataset, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    metric = ModelRandomizationMetric(model=model, train_dataset=dataset, explain_fn=lambda x: x, seed=42, device="cpu")
    rand_model = metric.rand_model
    for param1, param2 in zip(model.parameters(), rand_model.parameters()):
        assert not torch.allclose(param1.data, param2.data), "Test failed."
