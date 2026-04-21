import pytest
import torch

from quanda.explainers.wrappers import (
    CaptumSimilarity,
    captum_similarity_explain,
)
from quanda.utils.functions import cosine_similarity, dot_product_similarity


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset,  explanations, test_data, batch_size, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_samples_1",
            4,
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_samples_1",
            3,
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_explanations_dot_similarity_1",
            "load_mnist_test_samples_1",
            8,
            {"layers": "relu_4", "similarity_metric": dot_product_similarity},
        ),
    ],
)
def test_captum_similarity_explain(
    test_id,
    model,
    checkpoint,
    dataset,
    explanations,
    test_data,
    batch_size,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    explanations_exp = request.getfixturevalue(explanations)

    explainer = CaptumSimilarity(
        model=model,
        checkpoints=checkpoint,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        batch_size=batch_size,
        device="cpu",
        **method_kwargs,
    )

    explanations = explainer.explain(test_data)
    assert torch.allclose(explanations, explanations_exp), (
        "Training data attributions are not as expected"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset,  explanations, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_captum_similarity_self_influence(
    test_id,
    model,
    checkpoint,
    dataset,
    explanations,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)

    explainer = CaptumSimilarity(
        model=model,
        checkpoints=checkpoint,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )

    self_influence = explainer.self_influence()
    assert self_influence.shape[0] == len(dataset), (
        "Self influence attributions are not as expected"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint, dataset, test_data, "
    "explanations, method_kwargs",
    [
        (
            "mnist_dict_test_data",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_explanations_similarity_1",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_captum_similarity_explain_dict_test_data(
    test_id,
    model,
    checkpoint,
    dataset,
    test_data,
    explanations,
    method_kwargs,
    request,
    tmp_path,
):
    """Passing ``test_data`` as a dict must match the tensor path."""
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_data)
    explanations_exp = request.getfixturevalue(explanations)

    explainer = CaptumSimilarity(
        model=model,
        checkpoints=checkpoint,
        model_id="test_id_dict",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        batch_size=4,
        device="cpu",
        **method_kwargs,
    )

    got = explainer.explain({"x": test_tensor})
    assert torch.allclose(got, explanations_exp), (
        "Dict-valued test_data did not match the tensor path"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, method_kwargs, explanations",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
            "load_mnist_explanations_similarity_1",
        ),
    ],
)
def test_captum_similarity_explain_functional(
    test_id,
    model,
    checkpoint,
    dataset,
    test_data,
    method_kwargs,
    explanations,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    explanations_exp = request.getfixturevalue(explanations)
    explanations = captum_similarity_explain(
        model=model,
        checkpoints=checkpoint,
        model_id="test_id",
        cache_dir=str(tmp_path),
        test_data=test_data,
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), (
        "Training data attributions are not as expected"
    )
