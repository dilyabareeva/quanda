import pytest
import torch
from captum.influence import TracInCP, TracInCPFast, TracInCPFastRandProj
from captum.influence._core.arnoldi_influence_function import (  # type: ignore
    ArnoldiInfluenceFunction,
)
from torch.utils.data import TensorDataset

from quanda.explainers.wrappers import (
    CaptumArnoldi,
    CaptumSimilarity,
    CaptumTracInCP,
    CaptumTracInCPFast,
    CaptumTracInCPFastRandProj,
    captum_arnoldi_explain,
    captum_arnoldi_self_influence,
    captum_similarity_explain,
    captum_tracincp_explain,
    captum_tracincp_fast_explain,
    captum_tracincp_fast_rand_proj_explain,
    captum_tracincp_fast_self_influence,
    captum_tracincp_self_influence,
)
from quanda.utils.common import get_load_state_dict_func
from quanda.utils.functions import cosine_similarity, dot_product_similarity


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset,  explanations, test_tensor, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_samples_1",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist_prediction",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_explanations_dot_similarity_1",
            "load_mnist_test_samples_1",
            {"layers": "relu_4", "similarity_metric": dot_product_similarity},
        ),
    ],
)
def test_captum_similarity_explain(test_id, model, dataset, explanations, test_tensor, method_kwargs, request, tmp_path):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
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
    "test_id, model, dataset,  explanations, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_captum_similarity_self_influence(test_id, model, dataset, explanations, method_kwargs, request, tmp_path):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations_exp = request.getfixturevalue(explanations)

    explainer = CaptumSimilarity(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )

    self_influence = explainer.self_influence()
    assert self_influence.shape[0] == len(dataset), "Self influence attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, method_kwargs, explanations",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
            "load_mnist_explanations_similarity_1",
        ),
    ],
)
def test_captum_influence_explain_functional(
    test_id, model, dataset, test_tensor, method_kwargs, explanations, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    explanations_exp = request.getfixturevalue(explanations)
    explanations = captum_similarity_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        test_tensor=test_tensor,
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, explanation_targets,  method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {"batch_size": 1, "projection_dim": 10, "arnoldi_dim": 10},
        ),
        (
                "mnist",
                "load_mnist_model",
                "load_mnist_dataset",
                "load_mnist_test_samples_1",
                None,
                {"batch_size": 1, "projection_dim": 10, "arnoldi_dim": 10},
        ),
        (
            "mnist_prediction",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "batch_size": 1,
                "projection_dim": 10,
                "arnoldi_dim": 20,
                "arnoldi_tol": 2e-1,
                "hessian_reg": 2e-3,
                "hessian_inverse_tol": 2e-4,
                "projection_on_cpu": True,
            },
        ),
    ],
)
def test_captum_arnoldi(test_id, model, dataset, test_tensor, explanation_targets, method_kwargs, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    if explanation_targets is not None:
        explanation_targets = request.getfixturevalue(explanation_targets)

    explainer = CaptumArnoldi(
        model=model,
        train_dataset=dataset,
        checkpoint="tests/assets/mnist",
        device="cpu",
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs,
    )
    explanations = explainer.explain(test_tensor, explanation_targets)

    explainer_captum = ArnoldiInfluenceFunction(
        model=model,
        train_dataset=dataset,
        checkpoint="tests/assets/mnist",
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs,
    )
    if explanation_targets is None:
        explanations_captum = explainer_captum.influence(inputs=(test_tensor, model(test_tensor).argmax(dim=1)))
    else:
        explanations_captum = explainer_captum.influence(inputs=(test_tensor, explanation_targets))
    assert torch.allclose(explanations, explanations_captum), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, explanation_targets, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "batch_size": 1,
                "projection_dim": 10,
                "arnoldi_dim": 20,
                "arnoldi_tol": 1e-1,
                "hessian_reg": 1e-3,
                "hessian_inverse_tol": 1e-4,
                "projection_on_cpu": True,
            },
        ),
        (
            "mnist_prediction",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "batch_size": 1,
                "seed": 42,
                "projection_dim": 10,
                "arnoldi_dim": 20,
                "arnoldi_tol": 1e-1,
                "hessian_reg": 1e-3,
                "hessian_inverse_tol": 1e-4,
                "projection_on_cpu": True,
            },
        ),
    ],
)
def test_captum_arnoldi_explain_functional(
    test_id, model, dataset, test_tensor, explanation_targets, method_kwargs, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    if explanation_targets is not None:
        explanation_targets = request.getfixturevalue(explanation_targets)
    hessian_dataset = torch.utils.data.Subset(dataset, [0, 1])

    explainer_captum = ArnoldiInfluenceFunction(
        model=model,
        train_dataset=dataset,
        checkpoint="tests/assets/mnist",
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        test_loss_fn=torch.nn.NLLLoss(reduction="none"),
        hessian_dataset=hessian_dataset,
        **method_kwargs,
    )
    explanations_exp = explainer_captum.influence(inputs=(test_tensor, explanation_targets))

    explanations = captum_arnoldi_explain(
        model=model,
        test_tensor=test_tensor,
        train_dataset=dataset,
        explanation_targets=explanation_targets,
        device="cpu",
        checkpoint="tests/assets/mnist",
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        test_loss_fn=torch.nn.NLLLoss(reduction="none"),
        hessian_dataset=hessian_dataset,
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            {
                "batch_size": 1,
                "seed": 42,
                "projection_dim": 10,
                "arnoldi_dim": 20,
                "arnoldi_tol": 1e-1,
                "hessian_reg": 1e-3,
                "hessian_inverse_tol": 1e-4,
                "projection_on_cpu": True,
            },
        ),
    ],
)
def test_captum_arnoldi_self_influence(test_id, model, dataset, method_kwargs, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)

    explainer_captum = ArnoldiInfluenceFunction(
        model=model,
        train_dataset=dataset,
        checkpoint="tests/assets/mnist",
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs,
    )
    explanations_exp = explainer_captum.self_influence()

    explanations = captum_arnoldi_self_influence(
        model=model,
        train_dataset=dataset,
        device="cpu",
        checkpoint="tests/assets/mnist",
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, explanation_targets, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "get_mnist_checkpoints",
            {
                "batch_size": 1,
                "sample_wise_grads_per_batch": False,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="none"),
            },
        ),
        (
                "mnist_prediction",
                "load_mnist_model",
                "load_mnist_dataset",
                "load_mnist_test_samples_1",
                None,
                "get_mnist_checkpoints",
                {
                    "batch_size": 1,
                    "sample_wise_grads_per_batch": False,
                    "loss_fn": torch.nn.CrossEntropyLoss(reduction="none"),
                },
        ),
    ],
)
def test_captum_tracincp(test_id, model, dataset, test_tensor, explanation_targets, checkpoints, method_kwargs, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    if explanation_targets is not None:
        explanation_targets = request.getfixturevalue(explanation_targets)
    checkpoints = request.getfixturevalue(checkpoints)

    model.load_state_dict(torch.load(checkpoints[-1]))

    explainer_captum = TracInCP(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    if explanation_targets is None:
        explanations = explainer_captum.influence(inputs=(test_tensor, model(test_tensor).argmax(dim=1)))
    else:
        explanations = explainer_captum.influence(inputs=(test_tensor, explanation_targets))

    explainer = CaptumTracInCP(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    explanations_exp = explainer.explain(test_tensor, explanation_targets)

    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, checkpoints, test_tensor, explanation_targets, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "get_mnist_checkpoints",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "batch_size": 1,
                "sample_wise_grads_per_batch": False,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="none"),
                "test_loss_fn": torch.nn.NLLLoss(reduction="none"),
            },
        ),
    ],
)
def test_captum_tracincp_explain_functional(
    test_id, model, dataset, checkpoints, test_tensor, explanation_targets, method_kwargs, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    checkpoints = request.getfixturevalue(checkpoints)
    test_tensor = request.getfixturevalue(test_tensor)
    if explanation_targets is not None:
        explanation_targets = request.getfixturevalue(explanation_targets)

    explainer_captum = TracInCP(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations_exp = explainer_captum.influence(inputs=(test_tensor, explanation_targets))

    explanations = captum_tracincp_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        checkpoints=checkpoints,
        test_tensor=test_tensor,
        explanation_targets=explanation_targets,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "get_mnist_checkpoints",
            {
                "batch_size": 1,
                "sample_wise_grads_per_batch": False,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="none"),
                "layers": ["conv_1"],
            },
        ),
    ],
)
def test_captum_tracincp_self_influence(test_id, model, dataset, checkpoints, method_kwargs, request):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    checkpoints = request.getfixturevalue(checkpoints)

    explainer_captum = TracInCP(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations_exp = explainer_captum.self_influence(outer_loop_by_checkpoints=True)

    explanations = captum_tracincp_self_influence(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, explanation_targets, final_layer, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "model.fc_3",
            "get_mnist_checkpoints",
            {
                "batch_size": 1,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="sum"),
            },
        ),
        (
                "mnist_prediction",
                "load_mnist_model",
                "load_mnist_dataset",
                "load_mnist_test_samples_1",
                None,
                "model.fc_3",
                "get_mnist_checkpoints",
                {
                    "batch_size": 1,
                    "loss_fn": torch.nn.CrossEntropyLoss(reduction="sum"),
                },
        ),
    ],
)
def test_captum_tracincp_fast(
    test_id, model, dataset, test_tensor, explanation_targets, final_layer, checkpoints, method_kwargs, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    if explanation_targets is not None:
        explanation_targets = request.getfixturevalue(explanation_targets)
    checkpoints = request.getfixturevalue(checkpoints)

    model.load_state_dict(torch.load(checkpoints[-1]))

    final_fc_layer = model.fc_3

    explainer_captum = TracInCPFast(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    if explanation_targets is None:
        explanations = explainer_captum.influence(inputs=(test_tensor, model(test_tensor).argmax(dim=1)), k=None)
    else:
        explanations = explainer_captum.influence(inputs=(test_tensor, explanation_targets), k=None)

    explainer = CaptumTracInCPFast(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    explanations_exp = explainer.explain(test_tensor, explanation_targets)

    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, checkpoints, test_tensor, explanation_targets, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "get_mnist_checkpoints",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "batch_size": 1,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="sum"),
                "test_loss_fn": torch.nn.NLLLoss(reduction="sum"),
            },
        ),
    ],
)
def test_captum_tracincp_fast_explain_functional(
    test_id, model, dataset, checkpoints, test_tensor, explanation_targets, method_kwargs, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    checkpoints = request.getfixturevalue(checkpoints)
    test_tensor = request.getfixturevalue(test_tensor)
    if explanation_targets is not None:
        explanation_targets = request.getfixturevalue(explanation_targets)
    final_fc_layer = model.fc_3

    explainer_captum_simple = TracInCPFast(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations_exp_simple = explainer_captum_simple.influence(inputs=(test_tensor, explanation_targets), k=None)

    explanations_simple = captum_tracincp_fast_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        checkpoints=checkpoints,
        test_tensor=test_tensor,
        explanation_targets=explanation_targets,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        final_fc_layer=final_fc_layer,
        **method_kwargs,
    )
    assert torch.allclose(explanations_simple, explanations_exp_simple), "Training data attributions are not as expected"

    explainer_captum_complex = TracInCPFast(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations_exp_complex = explainer_captum_complex.influence(inputs=(test_tensor, explanation_targets), k=None)

    explanations_complex = captum_tracincp_fast_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        test_tensor=test_tensor,
        explanation_targets=explanation_targets,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    assert torch.allclose(explanations_complex, explanations_exp_complex), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "get_mnist_checkpoints",
            {
                "batch_size": 1,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="sum"),
            },
        ),
    ],
)
def test_captum_tracincp_fast_self_influence(test_id, model, dataset, checkpoints, method_kwargs, request, tmp_path):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
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
    explanations_exp = explainer_captum.self_influence(outer_loop_by_checkpoints=True)

    explanations = captum_tracincp_fast_self_influence(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        outer_loop_by_checkpoints=True,
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, explanation_targets, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "get_mnist_checkpoints",
            {
                "batch_size": 1,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="sum"),
                "projection_dim": 10,
                "seed": 42,
            },
        ),
        (
                "mnist_prediction",
                "load_mnist_model",
                "load_mnist_dataset",
                "load_mnist_test_samples_1",
                None,
                "get_mnist_checkpoints",
                {
                    "batch_size": 1,
                    "loss_fn": torch.nn.CrossEntropyLoss(reduction="sum"),
                    "projection_dim": 10,
                    "seed": 42,
                },
        ),
    ],
)
def test_captum_tracincp_fast_rand_proj(
    test_id, model, dataset, test_tensor, explanation_targets, checkpoints, method_kwargs, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    if explanation_targets is not None:
        explanation_targets = request.getfixturevalue(explanation_targets)
    checkpoints = request.getfixturevalue(checkpoints)
    final_fc_layer = model.fc_3

    explainer_captum = TracInCPFastRandProj(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    if explanation_targets is None:
        explanations = explainer_captum.influence(inputs=(test_tensor, model(test_tensor).argmax(dim=1)), k=None)
    else:
        explanations = explainer_captum.influence(inputs=(test_tensor, explanation_targets), k=None)

    explainer = CaptumTracInCPFastRandProj(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    explanations_exp = explainer.explain(test_tensor, explanation_targets)

    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, checkpoints, test_tensor, explanation_targets, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "get_mnist_checkpoints",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "batch_size": 1,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="sum"),
                "test_loss_fn": torch.nn.NLLLoss(reduction="sum"),
                "projection_dim": 10,
                "seed": 42,
            },
        ),
    ],
)
def test_captum_tracincp_fast_rand_proj_explain_functional(
    test_id, model, dataset, checkpoints, test_tensor, explanation_targets, method_kwargs, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    checkpoints = request.getfixturevalue(checkpoints)
    test_tensor = request.getfixturevalue(test_tensor)
    if explanation_targets is not None:
        explanation_targets = request.getfixturevalue(explanation_targets)
    final_fc_layer = model.fc_3

    explainer_captum_simple = TracInCPFastRandProj(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations_exp_simple = explainer_captum_simple.influence(inputs=(test_tensor, explanation_targets), k=None)

    explanations_simple = captum_tracincp_fast_rand_proj_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        checkpoints=checkpoints,
        test_tensor=test_tensor,
        explanation_targets=explanation_targets,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        final_fc_layer=final_fc_layer,
        **method_kwargs,
    )
    assert torch.allclose(explanations_simple, explanations_exp_simple), "Training data attributions are not as expected"

    explainer_captum_complex = TracInCPFastRandProj(
        model=model,
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations_exp_complex = explainer_captum_complex.influence(inputs=(test_tensor, explanation_targets), k=None)

    explanations_complex = captum_tracincp_fast_rand_proj_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        final_fc_layer=final_fc_layer,
        train_dataset=dataset,
        checkpoints=checkpoints,
        test_tensor=test_tensor,
        explanation_targets=explanation_targets,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    assert torch.allclose(explanations_complex, explanations_exp_complex), "Training data attributions are not as expected"
