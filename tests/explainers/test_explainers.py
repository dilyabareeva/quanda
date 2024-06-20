import os

import pytest
import torch

from src.explainers.functional import captum_similarity_explain
from src.explainers.wrappers.captum_influence import CaptumSimilarity
from src.utils.functions.similarities import cosine_similarity


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, method_kwargs, explanations",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
            "load_mnist_explanations_1",
        ),
    ],
)
def test_explain_functional(test_id, model, dataset, test_tensor, test_labels, method_kwargs, explanations, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    explanations_exp = request.getfixturevalue(explanations)
    explanations = captum_similarity_explain(
        model,
        "test_id",
        os.path.join("./cache", "test_id"),
        test_tensor,
        test_labels,
        dataset,
        device="cpu",
        init_kwargs=method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset,  explanations, test_tensor, test_labels, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_explanations_1",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_explain_stateful(test_id, model, dataset, explanations, test_tensor, test_labels, method_kwargs, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    explanations_exp = request.getfixturevalue(explanations)
    explainer = CaptumSimilarity(
        model=model,
        model_id="test_id",
        cache_dir=os.path.join("./cache", "test_id"),
        train_dataset=dataset,
        device="cpu",
        explainer_kwargs=method_kwargs,
    )
    explanations = explainer.explain(test_tensor)
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"
