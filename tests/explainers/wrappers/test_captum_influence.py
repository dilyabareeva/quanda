from collections import OrderedDict

import pytest
import torch
from torch.utils.data import TensorDataset

from src.explainers.wrappers.captum_influence import (
    CaptumSimilarity,
    captum_similarity_explain,
    captum_similarity_self_influence,
)
from src.utils.functions.similarities import (
    cosine_similarity,
    dot_product_similarity,
)


@pytest.mark.self_influence
@pytest.mark.parametrize(
    "test_id, init_kwargs",
    [
        (
            "random_data",
            {"layers": "identity", "similarity_metric": dot_product_similarity},
        ),
    ],
)
# TODO: I think a good naming convention is "test_<function_name>..." or "test_<class_name>...".
def test_self_influence(test_id, init_kwargs, tmp_path):
    # TODO: this should be a fixture.
    model = torch.nn.Sequential(OrderedDict([("identity", torch.nn.Identity())]))

    # TODO: those should be fixtures. We (most of the time) don't generate random data in tests.
    torch.random.manual_seed(42)
    X = torch.randn(100, 200)
    y = torch.randint(0, 10, (100,))
    rand_dataset = TensorDataset(X, y)

    # Using tmp_path pytest fixtures to create a temporary directory
    # TODO: One test should test one thing. This is test 1, ....
    self_influence_rank_functional = captum_similarity_self_influence(
        model=model,
        model_id="0",
        cache_dir=str(tmp_path),
        train_dataset=rand_dataset,
        init_kwargs=init_kwargs,
        device="cpu",
    ).argsort()

    # TODO: ...this is test 2, unless we want to compare that the outputs are the same.
    # TODO: If we want to test that the outputs are the same, we should have a separate test for that.
    explainer_obj = CaptumSimilarity(
        model=model,
        model_id="1",
        cache_dir=str(tmp_path),
        train_dataset=rand_dataset,
        device="cpu",
        **init_kwargs,
    )

    # TODO: self_influence is defined in BaseExplainer - there is a test in test_base_explainer for that.
    # TODO: here we then specifically test self_influence for CaptumSimilarity and should make it explicit in the name.
    self_influence_rank_stateful = explainer_obj.self_influence().argsort()

    # TODO: what if we pass a non-identity model? Then we don't expect torch.linalg.norm(X, dim=-1).argsort()
    # TODO: let's put expectations in the parametrisation of tests. We want to test different scenarios,
    #  and not some super-specific case. This specific case definitely can be tested as well.
    assert torch.allclose(self_influence_rank_functional, torch.linalg.norm(X, dim=-1).argsort())
    # TODO: I think it is best to stick to a single assertion per test (source: Google)
    assert torch.allclose(self_influence_rank_functional, self_influence_rank_stateful)


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
# TODO: I think a good naming convention is "test_<function_name>..." or "test_<class_name>...".
# TODO: I would call it test_captum_similarity, because it is a test for the CaptumSimilarity class.
# TODO: We could also make the explainer type (e.g. CaptumSimilarity) a param, then it would be test_explainer or something.
def test_explain_stateful(test_id, model, dataset, explanations, test_tensor, test_labels, method_kwargs, request, tmp_path):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    explanations_exp = request.getfixturevalue(explanations)

    explainer = CaptumSimilarity(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )

    explanations = explainer.explain(test_tensor)
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


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
def test_explain_functional(test_id, model, dataset, test_tensor, test_labels, method_kwargs, explanations, request, tmp_path):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    explanations_exp = request.getfixturevalue(explanations)
    explanations = captum_similarity_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        test_tensor=test_tensor,
        explanation_targets=test_labels,
        train_dataset=dataset,
        init_kwargs=method_kwargs,
        device="cpu",
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"