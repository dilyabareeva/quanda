from collections import OrderedDict

import pytest
import torch
from captum.influence import TracInCP

# TODO Should be imported directly from captum.influence once available
from captum.influence._core.arnoldi_influence_function import (  # type: ignore
    ArnoldiInfluenceFunction,
)
from torch.utils.data import TensorDataset

from quanda.explainers.wrappers import (
    CaptumArnoldi,
    CaptumSimilarity,
    CaptumTracInCP,
    captum_arnoldi_explain,
    captum_arnoldi_self_influence,
    captum_similarity_explain,
    captum_similarity_self_influence,
    captum_tracincp_explain,
    captum_tracincp_self_influence,
)
from quanda.utils.common import get_load_state_dict_func
from quanda.utils.functions import cosine_similarity, dot_product_similarity


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
        device="cpu",
        **init_kwargs,
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
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
# TODO: I think a good naming convention is "test_<function_name>..." or "test_<class_name>...".
# TODO: I would call it test_captum_similarity, because it is a test for the CaptumSimilarity class.
# TODO: We could also make the explainer type (e.g. CaptumSimilarity) a param, then it would be test_explainer or something.
def test_captum_influence_explain_stateful(
    test_id, model, dataset, explanations, test_tensor, test_labels, method_kwargs, request, tmp_path
):
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
            "load_mnist_explanations_similarity_1",
        ),
    ],
)
def test_captum_influence_explain_functional(
    test_id, model, dataset, test_tensor, test_labels, method_kwargs, explanations, request, tmp_path
):
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
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, method_kwargs_simple, method_kwargs_complex",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {"batch_size": 1, "projection_dim": 10, "arnoldi_dim": 10},
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
def test_captum_arnoldi(
    test_id, model, dataset, test_tensor, test_labels, method_kwargs_simple, method_kwargs_complex, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)

    explainer_simple = CaptumArnoldi(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        checkpoint="tests/assets/mnist",
        device="cpu",
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs_simple,
    )
    explanations_simple = explainer_simple.explain(test_tensor)

    explainer_captum_simple = ArnoldiInfluenceFunction(
        model=model,
        train_dataset=dataset,
        checkpoint="tests/assets/mnist",
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs_simple,
    )
    explanations_captum_simple = explainer_captum_simple.influence(inputs=(test_tensor, None))
    assert torch.allclose(explanations_simple, explanations_captum_simple), "Training data attributions are not as expected"

    explainer_complex = CaptumArnoldi(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        checkpoint="tests/assets/mnist",
        device="cpu",
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs_complex,
    )
    explanations_complex = explainer_complex.explain(test_tensor, test_labels)

    explainer_captum_complex = ArnoldiInfluenceFunction(
        model=model,
        train_dataset=dataset,
        checkpoint="tests/assets/mnist",
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs_complex,
    )
    explanations_captum_complex = explainer_captum_complex.influence(inputs=(test_tensor, test_labels))
    assert torch.allclose(explanations_complex, explanations_captum_complex), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, method_kwargs_simple, method_kwargs_complex",
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
    test_id, model, dataset, test_tensor, test_labels, method_kwargs_simple, method_kwargs_complex, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    hessian_dataset = torch.utils.data.Subset(dataset, [0, 1])

    explainer_captum_simple = ArnoldiInfluenceFunction(
        model=model,
        train_dataset=dataset,
        checkpoint="tests/assets/mnist",
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        sample_wise_grads_per_batch=False,
        **method_kwargs_simple,
    )
    explanations_exp_simple = explainer_captum_simple.influence(inputs=(test_tensor, None))

    explanations_simple = captum_arnoldi_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        test_tensor=test_tensor,
        train_dataset=dataset,
        device="cpu",
        checkpoint="tests/assets/mnist",
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        sample_wise_grads_per_batch=False,
        **method_kwargs_simple,
    )
    assert torch.allclose(explanations_simple, explanations_exp_simple), "Training data attributions are not as expected"

    explainer_captum_complex = ArnoldiInfluenceFunction(
        model=model,
        train_dataset=dataset,
        checkpoint="tests/assets/mnist",
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        loss_fn=torch.nn.CrossEntropyLoss(),
        test_loss_fn=torch.nn.NLLLoss(),
        hessian_dataset=hessian_dataset,
        sample_wise_grads_per_batch=True,
        **method_kwargs_complex,
    )
    explanations_exp_complex = explainer_captum_complex.influence(inputs=(test_tensor, test_labels))

    explanations_complex = captum_arnoldi_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        test_tensor=test_tensor,
        train_dataset=dataset,
        explanation_targets=test_labels,
        device="cpu",
        checkpoint="tests/assets/mnist",
        loss_fn=torch.nn.CrossEntropyLoss(),
        test_loss_fn=torch.nn.NLLLoss(),
        hessian_dataset=hessian_dataset,
        sample_wise_grads_per_batch=True,
        **method_kwargs_complex,
    )
    assert torch.allclose(explanations_complex, explanations_exp_complex), "Training data attributions are not as expected"


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
def test_captum_arnoldi_self_influence(test_id, model, dataset, method_kwargs, request, tmp_path):
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
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        device="cpu",
        checkpoint="tests/assets/mnist",
        loss_fn=torch.nn.CrossEntropyLoss(reduction="none"),
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, checkpoints, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "get_mnist_checkpoints",
            {
                "batch_size": 1,
                "sample_wise_grads_per_batch": False,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="none"),
            },
        ),
    ],
)
def test_captum_tracincp(test_id, model, dataset, test_tensor, checkpoints, method_kwargs, request, tmp_path):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    checkpoints = request.getfixturevalue(checkpoints)

    explainer_captum = TracInCP(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations = explainer_captum.influence(inputs=(test_tensor, None))

    explainer = CaptumTracInCP(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    explanations_exp = explainer.explain(test_tensor)

    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, checkpoints, test_tensor, test_labels, method_kwargs",
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
    test_id, model, dataset, checkpoints, test_tensor, test_labels, method_kwargs, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    checkpoints = request.getfixturevalue(checkpoints)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)

    explainer_captum_simple = TracInCP(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations_exp_simple = explainer_captum_simple.influence(inputs=(test_tensor, None))

    explanations_simple = captum_tracincp_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        checkpoints=checkpoints,
        test_tensor=test_tensor,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        **method_kwargs,
    )
    assert torch.allclose(explanations_simple, explanations_exp_simple), "Training data attributions are not as expected"

    explainer_captum_complex = TracInCP(
        model=model,
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        **method_kwargs,
    )
    explanations_exp_complex = explainer_captum_complex.influence(inputs=(test_tensor, test_labels))

    explanations_complex = captum_tracincp_explain(
        model=model,
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        checkpoints=checkpoints,
        test_tensor=test_tensor,
        explanation_targets=test_labels,
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
                "sample_wise_grads_per_batch": False,
                "loss_fn": torch.nn.CrossEntropyLoss(reduction="none"),
                "layers": ["conv_1"],
            },
        ),
    ],
)
def test_captum_tracincp_self_influence(test_id, model, dataset, checkpoints, method_kwargs, request, tmp_path):
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
        model_id="test_id",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=get_load_state_dict_func("cpu"),
        device="cpu",
        outer_loop_by_checkpoints=True,
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"
