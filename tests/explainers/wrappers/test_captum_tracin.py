import gc

import pytest
import torch
from captum.influence import TracInCP, TracInCPFast

from quanda.explainers.wrappers import (
    CaptumTracInCP,
    CaptumTracInCPFast,
    CaptumTracInCPFastRandProj,
)
from quanda.utils.common import get_load_state_dict_func


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
    checkpoints = request.getfixturevalue(checkpoints)[:2]

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
    "test_id, model, dataset, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_checkpoints",
            {
                "batch_size": 1,
                "sample_wise_grads_per_batch": False,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="none"),
            },
        ),
    ],
)
def test_captum_tracincp_self_influence(
    test_id,
    model,
    dataset,
    checkpoints,
    method_kwargs,
    request,
    mocker,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    checkpoints = request.getfixturevalue(checkpoints)[:2]
    n_train = len(dataset)

    mocker.patch.object(CaptumTracInCP, "_init_explainer")
    explainer = CaptumTracInCP(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    explainer.captum_explainer = mocker.MagicMock()
    explainer.captum_explainer.self_influence.return_value = torch.randn(
        n_train,
    )
    self_influence = explainer.self_influence()

    explainer.captum_explainer.self_influence.assert_called_once()
    assert self_influence.shape == (n_train,)


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
    checkpoints = request.getfixturevalue(checkpoints)[:2]
    final_fc_layer = model.fc_3

    n_test = len(test_data)
    n_train = len(dataset)

    mocker.patch.object(CaptumTracInCPFastRandProj, "_init_explainer")
    explainer = CaptumTracInCPFastRandProj(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    explainer.captum_explainer = mocker.MagicMock()
    explainer.captum_explainer.influence.return_value = torch.randn(
        n_test, n_train
    )
    explanations = explainer.explain(test_data, test_labels)

    explainer.captum_explainer.influence.assert_called_once()
    assert explanations.shape == (n_test, n_train)


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
    checkpoints = request.getfixturevalue(checkpoints)[:2]
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


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_checkpoints",
            {
                "batch_size": 1,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="sum"),
            },
        ),
    ],
)
def test_captum_tracincp_fast_self_influence(
    test_id,
    model,
    dataset,
    checkpoints,
    method_kwargs,
    request,
    mocker,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    checkpoints = request.getfixturevalue(checkpoints)[:2]
    final_fc_layer = model.fc_3
    n_train = len(dataset)

    mocker.patch.object(CaptumTracInCPFast, "_init_explainer")
    explainer = CaptumTracInCPFast(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    explainer.captum_explainer = mocker.MagicMock()
    explainer.captum_explainer.self_influence.return_value = torch.randn(
        n_train,
    )
    self_influence = explainer.self_influence()

    explainer.captum_explainer.self_influence.assert_called_once()
    assert self_influence.shape == (n_train,)
