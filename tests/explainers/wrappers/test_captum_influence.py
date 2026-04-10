import gc

import pytest
import torch
from captum.influence._core.arnoldi_influence_function import (  # type: ignore
    ArnoldiInfluenceFunction,
)

from quanda.explainers.wrappers import (
    CaptumArnoldi,
    captum_arnoldi_explain,
    captum_arnoldi_self_influence,
)
from quanda.utils.common import get_load_state_dict_func


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
