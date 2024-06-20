import os
import shutil
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
def test_self_influence(test_id, init_kwargs, request):
    # TODO: this should be a fixture.
    model = torch.nn.Sequential(OrderedDict([("identity", torch.nn.Identity())]))

    # TODO: those should be fixtures. We (most of the time) don't generate random data in tests.
    torch.random.manual_seed(42)
    X = torch.randn(100, 200)
    y = torch.randint(0, 10, (100,))
    rand_dataset = TensorDataset(X, y)

    # TODO: One test should test one thing. This is test 1, ....
    self_influence_rank_functional = captum_similarity_self_influence(
        model=model,
        model_id="0",
        cache_dir="temp_captum",
        train_dataset=rand_dataset,
        init_kwargs=init_kwargs,
        device="cpu",
    )

    # TODO: ...this is test 2, unless we want to compare that the outputs are the same.
    # TODO: If we want to test that the outputs are the same, we should have a separate test for that.
    explainer_obj = CaptumSimilarity(
        model=model,
        model_id="1",
        cache_dir="temp_captum2",
        train_dataset=rand_dataset,
        device="cpu",
        **init_kwargs,
    )

    # TODO: self_influence is defined in BaseExplainer - there is a test in test_base_explainer for that.
    # TODO: here we then specifically test self_influence for CaptumSimilarity and should make it explicit in the name.
    self_influence_rank_stateful = explainer_obj.self_influence()

    # TODO: we check "temp_captum2" but then remove os.path.join(os.getcwd(), "temp_captum2")?
    # TODO: is there a reason to fear that the "temp_captum2" folder is not in os.getcwd()?
    if os.path.isdir("temp_captum2"):
        shutil.rmtree(os.path.join(os.getcwd(), "temp_captum2"))
    if os.path.isdir("temp_captum"):
        shutil.rmtree(os.path.join(os.getcwd(), "temp_captum"))

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
        **method_kwargs,
    )
    # TODO: activations folder clean-up

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
