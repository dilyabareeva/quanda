import pytest
import torch

from quanda.explainers.wrappers import TRAK, trak_explain, trak_self_influence


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset,  explanations, test_tensor, test_labels, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_explanations_trak_1",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {"model_id": "0", "batch_size": 8, "seed": 42, "proj_dim": 10, "projector": "basic"},
        ),
    ],
)
# TODO: I think a good naming convention is "test_<function_name>..." or "test_<class_name>...".
# TODO: I would call it test_captum_similarity, because it is a test for the CaptumSimilarity class.
# TODO: We could also make the explainer type (e.g. CaptumSimilarity) a param, then it would be test_explainer or something.
def test_trak_wrapper_explain_stateful(
    test_id, model, dataset, explanations, test_tensor, test_labels, method_kwargs, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    explanations_exp = request.getfixturevalue(explanations)

    explainer = TRAK(model=model, cache_dir=tmp_path, train_dataset=dataset, **method_kwargs)

    explanations = explainer.explain(test=test_tensor, targets=test_labels)
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
            {"model_id": "0", "batch_size": 8, "seed": 42, "proj_dim": 10, "projector": "basic"},
            "load_mnist_explanations_trak_1",
        ),
    ],
)
def test_trak_wrapper_explain_functional(
    test_id, model, dataset, test_tensor, test_labels, method_kwargs, explanations, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    explanations_exp = request.getfixturevalue(explanations)
    explanations = trak_explain(
        model=model,
        cache_dir=str(tmp_path),
        test_tensor=test_tensor,
        train_dataset=dataset,
        explanation_targets=test_labels,
        device="cpu",
        **method_kwargs,
    )
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
            {"model_id": "0", "batch_size": 8, "seed": 42, "proj_dim": 10, "projector": "basic"},
            "load_mnist_explanations_trak_si_1",
        ),
    ],
)
def test_trak_wrapper_explain_self_influence_functional(
    test_id, model, dataset, test_tensor, test_labels, method_kwargs, explanations, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations_exp = request.getfixturevalue(explanations)
    explanations = trak_self_influence(
        model=model,
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"
