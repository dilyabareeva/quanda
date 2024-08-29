import os

import pytest
import torch
from trak import TRAKer
from trak.projectors import BasicProjector, CudaProjector, NoOpProjector

from quanda.explainers.wrappers import TRAK, trak_explain, trak_self_influence

projector_cls = {
    "cuda": CudaProjector,
    "basic": BasicProjector,
    "noop": NoOpProjector,
}


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {"model_id": "0", "batch_size": 8, "seed": 42, "proj_dim": 10, "projector": "basic"},
        ),
    ],
)
def test_trak(test_id, model, dataset, test_tensor, test_labels, method_kwargs, request, tmp_path):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)

    os.mkdir(str(tmp_path) + "/trak_0_cache")
    os.mkdir(str(tmp_path) + "/trak_1_cache")
    explainer = TRAK(model=model, cache_dir=str(tmp_path) + "/trak_0_cache", train_dataset=dataset, **method_kwargs)

    explanations = explainer.explain(test=test_tensor, targets=test_labels)

    batch_size = method_kwargs["batch_size"]
    ld = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    num_params_for_grad = 0
    params_iter = model.parameters()
    for p in list(params_iter):
        num_params_for_grad = num_params_for_grad + p.numel()

    projector_kwargs = {
        "grad_dim": num_params_for_grad,
        "proj_dim": method_kwargs["proj_dim"],
        "proj_type": "normal",
        "seed": method_kwargs["seed"],
        "device": "cpu",
    }
    projector = method_kwargs["projector"]
    projector_obj = projector_cls[projector](**projector_kwargs)

    traker = TRAKer(
        model=model,
        task="image_classification",
        train_set_size=explainer.dataset_length,
        projector=projector_obj,
        proj_dim=method_kwargs["proj_dim"],
        projector_seed=method_kwargs["seed"],
        save_dir=str(tmp_path) + "/trak_1_cache",
        device="cpu",
        use_half_precision=False,
    )
    traker.load_checkpoint(model.state_dict(), model_id=0)

    for i, (x, y) in enumerate(iter(ld)):
        traker.featurize(batch=(x, y), inds=torch.tensor([i * batch_size + j for j in range(x.shape[0])]))

    traker.finalize_features()
    if projector == "basic":
        # finalize_features frees memory so projector.proj_matrix needs to be reconstructed
        traker.projector = projector_cls[projector](**projector_kwargs)

    traker.start_scoring_checkpoint(
        model_id=0, checkpoint=model.state_dict(), exp_name="test", num_targets=test_tensor.shape[0]
    )
    traker.score(batch=(test_tensor, test_labels), num_samples=test_tensor.shape[0])
    explanations_exp = torch.from_numpy(traker.finalize_scores(exp_name="test")).T

    # torch.save(explanations, "tests/assets/mnist_test_suite_1/mnist_TRAK_tda.pt")
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {"model_id": "0", "batch_size": 8, "seed": 42, "proj_dim": 10, "projector": "basic"},
        ),
    ],
)
def test_trak_cache(test_id, model, dataset, test_tensor, test_labels, method_kwargs, request, tmp_path):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)

    explainer = TRAK(model=model, cache_dir=str(tmp_path), train_dataset=dataset, **method_kwargs)

    explanations = explainer.explain(test=test_tensor, targets=test_labels)
    test_tensor = torch.ones_like(test_tensor)
    explanations_2 = explainer.explain(test=test_tensor, targets=test_labels)
    assert not torch.allclose(explanations, explanations_2), "Caching is problematic inside the lifetime of the wrapper"


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
def test_trak_explain_functional(
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
        **method_kwargs,
    )
    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {"model_id": "0", "batch_size": 8, "seed": 42, "proj_dim": 10, "projector": "basic"},
        ),
    ],
)
def test_trak_explain_functional_cache(test_id, model, dataset, test_tensor, test_labels, method_kwargs, request, tmp_path):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    explanations_first = trak_explain(
        model=model,
        cache_dir=str(tmp_path),
        test_tensor=test_tensor,
        train_dataset=dataset,
        explanation_targets=test_labels,
        **method_kwargs,
    )
    test_tensor = torch.rand_like(test_tensor)
    explanations_second = trak_explain(
        model=model,
        cache_dir=str(tmp_path),
        test_tensor=test_tensor,
        train_dataset=dataset,
        explanation_targets=test_labels,
        **method_kwargs,
    )
    assert not torch.allclose(
        explanations_first, explanations_second
    ), "Caching is problematic between different instantiations"


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
def test_trak_self_influence_functional(
    test_id, model, dataset, test_tensor, test_labels, method_kwargs, explanations, request, tmp_path
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations_exp = request.getfixturevalue(explanations)
    explanations = trak_self_influence(
        model=model,
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        **method_kwargs,
    )

    assert torch.allclose(explanations, explanations_exp), "Training data attributions are not as expected"
