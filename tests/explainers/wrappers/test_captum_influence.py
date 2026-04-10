import gc
import os

import pytest
import torch
from captum.influence import TracInCP, TracInCPFast
from captum.influence._core.arnoldi_influence_function import (  # type: ignore
    ArnoldiInfluenceFunction,
)

from quanda.explainers.wrappers import (
    CaptumArnoldi,
    CaptumSimilarity,
    CaptumTracInCP,
    CaptumTracInCPFast,
    CaptumTracInCPFastRandProj,
    captum_arnoldi_explain,
    captum_arnoldi_self_influence,
    captum_similarity_explain,
)
from quanda.utils.common import get_load_state_dict_func
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


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, test_labels, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {"batch_size": 1, "projection_dim": 5, "arnoldi_dim": 5},
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "batch_size": 1,
                "projection_dim": 5,
                "arnoldi_dim": 10,
                "arnoldi_tol": 2e-1,
                "hessian_reg": 2e-3,
                "hessian_inverse_tol": 2e-4,
                "projection_on_cpu": True,
            },
        ),
    ],
)
def test_captum_arnoldi(
    test_id,
    model,
    checkpoint,
    dataset,
    test_data,
    test_labels,
    method_kwargs,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    test_labels = request.getfixturevalue(test_labels)

    explainer = CaptumArnoldi(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoint,
        device="cpu",
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs,
    )
    explanations = explainer.explain(test_data, test_labels)
    del explainer
    gc.collect()

    explainer_captum = ArnoldiInfluenceFunction(
        model=model,
        train_dataset=dataset,
        checkpoint=checkpoint,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs,
    )
    explanations_captum = explainer_captum.influence(
        inputs=(test_data, test_labels)
    )
    del explainer_captum
    gc.collect()

    assert torch.allclose(explanations, explanations_captum), (
        "Training data attributions are not as expected"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, test_labels, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "batch_size": 1,
                "projection_dim": 5,
                "arnoldi_dim": 10,
                "arnoldi_tol": 1e-1,
                "hessian_reg": 1e-3,
                "hessian_inverse_tol": 1e-4,
                "projection_on_cpu": True,
            },
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "batch_size": 1,
                "seed": 42,
                "projection_dim": 5,
                "arnoldi_dim": 10,
                "arnoldi_tol": 1e-1,
                "hessian_reg": 1e-3,
                "hessian_inverse_tol": 1e-4,
                "projection_on_cpu": True,
            },
        ),
    ],
)
def test_captum_arnoldi_explain_functional(
    test_id,
    model,
    checkpoint,
    dataset,
    test_data,
    test_labels,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    test_labels = request.getfixturevalue(test_labels)
    hessian_dataset = torch.utils.data.Subset(dataset, [0, 1])

    explainer_captum = ArnoldiInfluenceFunction(
        model=model,
        train_dataset=dataset,
        checkpoint=checkpoint,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        test_loss_fn=torch.nn.NLLLoss(reduction="none"),
        hessian_dataset=hessian_dataset,
        **method_kwargs,
    )
    explanations_exp = explainer_captum.influence(
        inputs=(test_data, test_labels)
    )
    del explainer_captum
    gc.collect()

    explanations = captum_arnoldi_explain(
        model=model,
        checkpoints=checkpoint,
        test_data=test_data,
        train_dataset=dataset,
        explanation_targets=test_labels,
        device="cpu",
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        test_loss_fn=torch.nn.NLLLoss(reduction="none"),
        hessian_dataset=hessian_dataset,
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), (
        "Training data attributions are not as expected"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            {
                "batch_size": 1,
                "seed": 42,
                "projection_dim": 5,
                "arnoldi_dim": 10,
                "arnoldi_tol": 1e-1,
                "hessian_reg": 1e-3,
                "hessian_inverse_tol": 1e-4,
                "projection_on_cpu": True,
            },
        ),
    ],
)
def test_captum_arnoldi_self_influence(
    test_id, model, checkpoint, dataset, method_kwargs, request
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)

    explainer_captum = ArnoldiInfluenceFunction(
        model=model,
        train_dataset=dataset,
        checkpoint=checkpoint,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs,
    )
    explanations_exp = explainer_captum.self_influence()
    del explainer_captum
    gc.collect()

    explanations = captum_arnoldi_self_influence(
        model=model,
        train_dataset=dataset,
        device="cpu",
        checkpoints=checkpoint,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), (
        "Training data attributions are not as expected"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_data, test_lables, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "load_mnist_checkpoints",
            {
                "batch_size": 1,
                "sample_wise_grads_per_batch": False,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="none"),
            },
        ),
    ],
)
def test_captum_tracincp(
    test_id,
    model,
    dataset,
    test_data,
    test_lables,
    checkpoints,
    method_kwargs,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    test_lables = request.getfixturevalue(test_lables)
    checkpoints = request.getfixturevalue(checkpoints)

    explainer_captum = TracInCP(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations = explainer_captum.influence(inputs=(test_data, test_lables))
    del explainer_captum
    gc.collect()

    explainer = CaptumTracInCP(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    explanations_exp = explainer.explain(test_data, test_lables)

    assert torch.allclose(explanations, explanations_exp), (
        "Training data attributions are not as expected"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, test_labels, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "load_mnist_checkpoints",
            {
                "batch_size": 1,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="sum"),
            },
        ),
    ],
)
def test_captum_tracincp_fast(
    test_id,
    model,
    checkpoint,
    dataset,
    test_data,
    test_labels,
    checkpoints,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    test_labels = request.getfixturevalue(test_labels)
    checkpoints = request.getfixturevalue(checkpoints)
    final_fc_layer = model.fc_3

    explainer_captum = TracInCPFast(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations = explainer_captum.influence(
        inputs=(test_data, test_labels), k=None
    )
    del explainer_captum
    gc.collect()

    explainer = CaptumTracInCPFast(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    explanations_exp = explainer.explain(test_data, test_labels)

    assert torch.allclose(explanations, explanations_exp), (
        "Training data attributions are not as expected"
    )


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, test_labels, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "load_mnist_checkpoints",
            {
                "batch_size": 1,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="sum"),
                "projection_dim": 5,
                "seed": 42,
            },
        ),
    ],
)
def test_captum_tracincp_fast_rand_proj(
    test_id,
    model,
    checkpoint,
    dataset,
    test_data,
    test_labels,
    checkpoints,
    method_kwargs,
    request,
    tmp_path,
    mocker,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    test_labels = request.getfixturevalue(test_labels)
    checkpoints = request.getfixturevalue(checkpoints)
    final_fc_layer = model.fc_3

    n_test = len(test_data)
    n_train = len(dataset)
    mock_influence = mocker.patch(
        "captum.influence.TracInCPFastRandProj.influence",
        return_value=torch.randn(n_test, n_train),
    )

    explainer = CaptumTracInCPFastRandProj(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    explanations = explainer.explain(test_data, test_labels)

    mock_influence.assert_called_once()
    assert explanations.shape == (n_test, n_train)
