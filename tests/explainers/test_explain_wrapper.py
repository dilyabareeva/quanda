import os

import pytest
import torch

from explainers.explain_wrapper import explain


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, method, method_kwargs, explanations",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "SimilarityInfluence",
            {"layer": "relu_4"},
            "load_mnist_explanations_1",
        ),
    ],
)
def test_explain(test_id, model, dataset, explanations, test_tensor, method, method_kwargs, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    explanations_exp = request.getfixturevalue(explanations)
    explanations = explain(
        model,
        test_id,
        os.path.join("./cache", "test_id"),
        dataset,
        test_tensor,
        method,
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"
